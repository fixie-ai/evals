import base64
import dataclasses
import io
import logging
import queue
import threading
import time
import traceback
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, TypeVar

import librosa
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from evals.solvers.solver import Solver, SolverResult
from evals.solvers.utils import BatchedProcessPoolExecutor
from evals.task_state import TaskState

SAMPLE_RATE = 16000
DEFAULT_MAX_BATCH_SIZE = 32


class FixieGPUSolver(Solver):
    """
    A solver class for the locally running a model on multiple GPUs using a ProcessPoolExecutor.
    The model is employed using a Hugging Face Transformers pipeline. We assume that the pipeline
    is an UltravoxPipeline class and the inputs are text/audio messages.

    The way that it works is that
    1. Initiatization:
        a. We create `num_gpus` processes, each with a copy of the model.
        b. Process 0 is run first to download the model and avoid race conditions.
        c. The other processes are started after the model is downloaded.

    2. Processing:
        a. We need to use a high enough `EVALS_THREADS` count to flood the requests into the requests queue.
            I recommend using at least `2 * MAX_BATCH_SIZE * num_gpus` threads.
        b. BatchedProcessPoolExecutor gathers as many samples as are available for its batch to submit for inference.
            A maximum batch size is enforced.
        c. Batch inference is done after preprocessing and collating the inputs.
        d. Resulting completions are returned to the right threads by BatchedProcessPoolExecutor.

    Completion function parameters:
    - model: str - The model name/path to use for the pipeline
    - num_gpus: int - The number of GPUs to use for parallel processing (default: all available GPUs)
    - max_batch_size: int - The maximum batch size to use for inference (default: 32)
    - extra_options: Dict[str, Any] - Extra options to pass to the pipeline
        - max_new_tokens: int - The maximum number of new tokens to generate in a single call (default: 256)
        - temperature: float - The temperature to use for sampling (default: 1.0)
        - repetition_penalty: float - The repetition penalty to use for sampling (default: 1.0)
    """

    def __init__(
        self,
        model: str,
        num_gpus: int = torch.cuda.device_count(),
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)

        if extra_options is None:
            extra_options = {}

        # Set the start method for the entire script
        mp.set_start_method("spawn")

        rank_queue = mp.Queue()

        # First, we only start the primary process/GPU to let it download the model and avoid race-conditions
        rank_queue.put(0)

        if "max_new_tokens" not in extra_options:
            extra_options["max_new_tokens"] = 256

        print("Using batch size: \n", max_batch_size)

        self.executor = BatchedProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            max_batch_size=int(max_batch_size),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model, extra_options),
            batch_worker_fn=solver_worker,
        )

    def copy(self):
        # The queue objects (in self.executor) must not be copied
        return self

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        inputs = {}
        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        if not isinstance(msgs[-1]["content"], str):
            # This means the last message is an audio message
            parts = msgs[-1]["content"]
            parts_str = [x["text"] if x["type"] == "text" else "<|audio|>" for x in parts]
            # Concatenate all text parts into a single string
            msgs[-1]["content"] = "".join(parts_str)
            data_parts = [x["audio_url"] for x in parts if x["type"] == "audio_url"]
            assert len(data_parts) == 1
            # Extract the audio data from the last message
            audio_data = data_parts[0]["url"].split(",")[1]
            audio_stream = io.BytesIO(base64.b64decode(audio_data))

            # Read the audio data using soundfile and enforce the expected sample rate
            inputs["audio"] = librosa.load(audio_stream, sr=SAMPLE_RATE)[0]
            inputs["sampling_rate"] = SAMPLE_RATE

        inputs["turns"] = msgs

        # This is where the magic happens: we send the inputs to be processed by the model
        completion_output = self.executor.submit(inputs).result()

        solver_result = SolverResult(completion_output)

        return solver_result

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        audio_values = [f.pop("audio_values", None) for f in features]
        input_ids_lens = torch.LongTensor([f["input_ids"].shape[-1] for f in features])
        batch = super().__call__(features, *args, **kwargs)

        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values and audio_values[0] is not None:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.stack(
                [F.pad(x, (0, max_len - x.shape[-1])) for x in audio_values]
            )
            if self.tokenizer.padding_side == "left":
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )

        return batch


def solver_initializer(
    rank_queue: mp.Queue, world_size: int, model: str, extra_options: Dict[str, Any]
):
    import os
    import logging
    
    rank = rank_queue.get()
    
    # Only initialize distributed setup if we have multiple GPUs
    use_distributed = world_size > 1
    if use_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        try:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=1)
            )
        except Exception as e:
            logging.warning(f"Failed to initialize distributed process group: {e}")
            use_distributed = False

    global pipe, collator

    config = transformers.AutoConfig.from_pretrained(
        model,
        trust_remote_code=True
    )
    
    # Initialize model with Accelerate for efficient multi-GPU inference
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )
    
    # Load and dispatch model across available GPUs
    model = load_checkpoint_and_dispatch(
        model,
        model,
        device_map="auto",
        dtype=torch.bfloat16,
        no_split_module_classes=["GPTBigCodeBlock", "LlamaDecoderLayer"]  # Adjust based on your model
    )

    pipe = transformers.pipeline(
        "ultravox-pipeline",
        model=model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **extra_options,
    )

    pipe.tokenizer.padding_side = "left"
    pipe.processor = transformers.AutoProcessor.from_pretrained(model)

    collator = DataCollatorForSeq2SeqWithAudio(tokenizer=pipe.tokenizer)

    if use_distributed:
        try:
            torch.distributed.barrier()
        except Exception as e:
            logging.warning(f"Failed to synchronize at barrier: {e}")

    if rank == 0:
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(inputs: List[Dict[str, Any]]):
    prepped = [pipe.preprocess(item) for item in inputs]
    prepped = [
        {k: v.to(pipe.model.device) for k, v in sample.items()} 
        for sample in prepped
    ]

    batch = collator(prepped)
    batch = {k: v.to(pipe.model.device) for k, v in batch.items() if v is not None}

    with torch.inference_mode():
        terminators = [pipe.tokenizer.eos_token_id]
        if "<|eot_id|>" in pipe.tokenizer.added_tokens_encoder:
            terminators.append(pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        input_len = batch["input_ids"].shape[1]

        # The model will automatically handle tensor parallelism during generation
        outputs = pipe.model.generate(
            **batch,
            eos_token_id=terminators,
            **pipe._forward_params,
        )
        out_texts = [
            pipe.tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs
        ]
        print(f"Generated outputs: {out_texts}")
        return out_texts

