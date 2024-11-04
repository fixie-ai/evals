import base64
import dataclasses
import io
import logging
import queue
import threading
import time
import traceback
import torch
import torch.distributed
import torch.multiprocessing as mp
import librosa
import transformers
from urllib.request import urlopen
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

SAMPLE_RATE = 16000
DEFAULT_MAX_BATCH_SIZE = 32

class DivaLocalGPUSolver(Solver):
    """
    A solver class for running the DiVA model in parallel across multiple GPUs.
    Uses BatchedProcessPoolExecutor for efficient batch processing.

    Args:
        model_name: str - The model name/path (default: "WillHeld/DiVA-llama-3-v0-8b")
        num_gpus: int - Number of GPUs to use (default: all available)
        max_batch_size: int - Maximum batch size for inference
        extra_options: Dict[str, Any] - Additional options for model generation
        postprocessors: list[str] - List of postprocessors to apply
        registry: Any - Registry object for the solver
    """

    def __init__(
        self,
        model_name: str,
        num_gpus: int = torch.cuda.device_count(),
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)

        self.model_name = model_name
        self.extra_options = extra_options or {}

        # Set up multiprocessing
        mp.set_start_method("spawn", force=True)
        rank_queue = mp.Queue()
        rank_queue.put(0)  # Start with primary GPU

        self.executor = BatchedProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            max_batch_size=int(max_batch_size),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model_name),
            batch_worker_fn=solver_worker,
        )

    def copy(self):
        return self

    def _process_audio_content(self, content: list) -> tuple[list, list]:
        """Process audio content from message parts."""
        audios = []
        prompts = []
        
        for part in content:
            if part["type"] == "audio_url":
                if isinstance(part["audio_url"], dict) and "url" in part["audio_url"]:
                    audio_data = part["audio_url"]["url"].split(",")[1]
                    audio_stream = io.BytesIO(base64.b64decode(audio_data))
                else:
                    audio_stream = io.BytesIO(urlopen(part["audio_url"]).read())
                
                audio = librosa.load(audio_stream, sr=SAMPLE_RATE)[0]
                audios.append(audio)
                
            elif part["type"] == "text":
                prompts.append(part["text"])
                
        return audios, prompts

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        inputs = {"audios": [], "prompts": []}
        
        # Process the last message if it contains audio
        if not isinstance(task_state.messages[-1].content, str):
            audios, prompts = self._process_audio_content(task_state.messages[-1].content)
            inputs["audios"].extend(audios)
            inputs["prompts"].extend(prompts) if prompts else inputs["prompts"].extend([None] * len(audios))
            
        # Submit to executor and get result
        completion_output = self.executor.submit(inputs).result()
        print("completion_output: \n", completion_output, "\n\n")
        return SolverResult(completion_output)

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


def solver_initializer(
    rank_queue: mp.Queue,
    world_size: int,
    model_name: str,
):
    """Initialize the model on the specified GPU."""
    rank = rank_queue.get()

    global model
    model = transformers.AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if rank == 0:
        # Let other initializers start after model is downloaded
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(inputs: List[Dict[str, Any]]) -> List[str]:
    """Process a batch of inputs using the model."""
    batch_audios = []
    batch_prompts = []
    
    # Process each input in the batch
    for input_item in inputs:
        batch_audios.extend(input_item["audios"])
        batch_prompts.extend(input_item["prompts"])

    # Generate responses using DiVA's interface
    responses = model.generate(
        batch_audios,
        batch_prompts if any(batch_prompts) else None
    )
    
    return responses



T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")

# Reuse the BatchedProcessPoolExecutor and related classes from the original implementation
@dataclasses.dataclass
class BatchableWorkItem:
    request: T_In
    future: futures.Future


class BatchedProcessPoolExecutor:
    def __init__(
        self,
        *args,
        batch_worker_fn: Callable[[List[T_In]], List[T_Out]],
        max_batch_size: int,
        max_workers: int = 1,
        **kwargs
    ):
        self.max_batch_size = max_batch_size
        self.batch_worker_fn = batch_worker_fn
        self._batch_queue = queue.Queue()
        self.available_workers = threading.Semaphore(value=max_workers + 1)
        self.process_pool_executor = ProcessPoolExecutor(
            *args, max_workers=max_workers, **kwargs
        )
        self._batch_thread = threading.Thread(target=self.batch_requests)
        self._batch_thread.start()

    def submit(self, request: T_In) -> futures.Future:
        item = BatchableWorkItem(request, futures.Future())
        self._batch_queue.put(item)
        return item.future

    def shutdown(self):
        self.process_pool_executor.shutdown()
        while not self._batch_queue.empty():
            try:
                item = self._batch_queue.get(block=False)
                if item is not None:
                    item.future.set_exception(Exception("The pool has already shut down."))
            except queue.Empty:
                pass
        self._batch_queue.put(None)

    def batch_requests(self):
        time.sleep(1)
        while True:
            self.available_workers.acquire()
            work_items: List[BatchableWorkItem] = [self._batch_queue.get()]
            
            while len(work_items) < self.max_batch_size:
                try:
                    item = self._batch_queue.get(block=False)
                    work_items.append(item)
                except queue.Empty:
                    break

            if work_items[-1] is None:
                if len(work_items) > 1:
                    logging.warn(
                        "There remained work items in the queue when shutting down. The items will be ignored."
                    )
                return

            requests = [item.request for item in work_items]
            task_futures = [item.future for item in work_items]

            try:
                result_future = self.process_pool_executor.submit(self.batch_worker_fn, requests)
            except Exception as e:
                self._handle_exception(e, task_futures)
                return

            result_future.add_done_callback(_set_results_cb(task_futures, self._handle_exception))
            result_future.add_done_callback(lambda _: self.available_workers.release())

    def _handle_exception(self, e: Exception, task_futures: List[futures.Future]):
        print(traceback.format_exc())
        for f in task_futures:
            if not f.done():
                f.set_exception(e)
        self.shutdown()


def _set_results_cb(task_futures: List[futures.Future], handle_exception_cb: Callable):
    def cb(batch_future: futures.Future):
        try:
            for f, r in zip(task_futures, batch_future.result()):
                f.set_result(r)
        except Exception as e:
            handle_exception_cb(e, task_futures)

    return cb