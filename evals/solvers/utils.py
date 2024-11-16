from typing import Union

import dataclasses
import logging
import queue
import threading
import time
import traceback
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, TypeVar

from evals.api import CompletionFn, DummyCompletionFn
from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.completion_fns.solver_completion_fn import SolverCompletionFn
from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import DummySolver, Solver


def maybe_wrap_with_compl_fn(ambiguous_executor: Union[CompletionFn, Solver]) -> CompletionFn:
    """
    Converts a solver into a completion function if it isn't already one.
    If it is already a completion function, it is returned unchanged.
    """
    if isinstance(ambiguous_executor, Solver):
        completion_fn = SolverCompletionFn(solver=ambiguous_executor)
    elif isinstance(ambiguous_executor, CompletionFn):
        completion_fn = ambiguous_executor
    else:
        raise ValueError(
            f"Expected `executor` to be a `CompletionFn` or `Solver`, "
            f"but got {ambiguous_executor}"
        )

    return completion_fn


def maybe_wrap_with_solver(ambiguous_executor: Union[Solver, CompletionFn]) -> Solver:
    """
    Converts a basic completion_fn into a Solver if it isn't already one.
    If it is already a Solver, it is returned unchanged.
    """

    if isinstance(ambiguous_executor, Solver):
        # Use the solver directly
        solver = ambiguous_executor
    elif isinstance(ambiguous_executor, SolverCompletionFn):
        # unwrap previously wrapped solver
        solver = ambiguous_executor.solver
    else:
        # Wrap the completion_fn in an appropriate solver for its type
        if isinstance(ambiguous_executor, OpenAIChatCompletionFn) or isinstance(
            ambiguous_executor, OpenAICompletionFn
        ):
            solver = OpenAISolver(
                completion_fn_options={
                    "model": ambiguous_executor.model,
                }
            )
            solver.completion_fn = ambiguous_executor
        elif isinstance(ambiguous_executor, DummyCompletionFn):
            solver = DummySolver()
        else:
            raise ValueError(f"Unsupported completion_fn type: {type(ambiguous_executor)}")
    return solver



T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


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
        # shut down the process pool executor
        self.process_pool_executor.shutdown()

        while not self._batch_queue.empty():
            try:
                item = self._batch_queue.get(block=False)
                if item is not None:
                    item.future.set_exception(Exception("The pool has already shut down."))
            except queue.Empty:
                pass

        # signal the batch thread to stop
        self._batch_queue.put(None)
        print("batched process pool executor has shut down.")

    def batch_requests(self):
        """
        Batches requests and dispatches them to the process pool executor

        Steps:
        1. greedily grab items
        2. dispatch them to ProcessPoolExecutor
        3. set the results back on the source future
        """
        # Wait a bit for the GPUs to be ready and allow the batch_queue to fill up a bit
        time.sleep(1)

        while True:
            # We don't wait to rush ahead too fast and fill up the queue with
            # batch_size=1 requests, but we also don't want any GPUs to be idle.
            self.available_workers.acquire()

            # greedily grab items
            work_items: List[BatchableWorkItem] = [self._batch_queue.get()]
            while len(work_items) < self.max_batch_size:
                try:
                    item = self._batch_queue.get(block=False)
                    work_items.append(item)
                except queue.Empty:
                    break

            # When we're done, a None item is added to the queue to signal the end of requests.
            # We will ignore any existing work_items since the process pool executor
            # is already shutting down.
            if work_items[-1] is None:
                if len(work_items) > 1:
                    logging.warn(
                        "There remained work items in the queue when shutting down. The items will be ignored."
                    )
                return

            requests = [item.request for item in work_items]
            task_futures = [item.future for item in work_items]

            # dispatch to the process pool
            try:
                result_future = self.process_pool_executor.submit(self.batch_worker_fn, requests)
            except Exception as e:
                self._handle_exception(e, task_futures)
                return

            # add callback for when the result is ready
            result_future.add_done_callback(_set_results_cb(task_futures, self._handle_exception))
            result_future.add_done_callback(lambda _: self.available_workers.release())

    def _handle_exception(self, e: Exception, task_futures: List[futures.Future]):
        """
        Handles exceptions by simply panicking:
         * prints the traceback to allow debugging
         * sets exception on all the task futures
         * shuts down the executor
        """
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
