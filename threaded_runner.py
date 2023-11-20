import random
import threading
import time
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

Result = float
Index = int
Priority = Union[int, float]  # contains float so that we can use float("inf")
Time = float


class EndOfQueue:
    """
    End of queue sentinel.
    """

    ...


class Status(Enum):
    NOT_STARTED = "not started"
    IN_PROGRESS = "in progress"
    DONE = "done"
    FAILED = "failed"


class Task:
    def __init__(
        self,
        f: Callable[[], Any],
        index: Index,  # include index for logging purposes for now
    ):
        self._f = f
        self._index = index
        self._result: Optional[Result] = None
        self._status: Status = Status.NOT_STARTED
        self._lock = threading.Lock()
        self._num_attempts = 0

    def run(self) -> None:
        with self._lock:
            self._status = Status.IN_PROGRESS
            self._num_attempts += 1
            print(f"(Task {self._index}, Attempt {self._num_attempts}): {self._status.value}")
        try:
            output = self._f()
            time.sleep(output)
            if output < 0.3:  # simulate failure
                raise Exception
        except Exception:
            with self._lock:
                self._status = Status.FAILED
                print(f"(Task {self._index}, Attempt {self._num_attempts}): {self._status.value}")
            return
        with self._lock:
            self._result = random.random()
            self._status = Status.DONE
            print(f"(Task {self._index}, Attempt {self._num_attempts}): {self._status.value}")

    @property
    def result(self) -> Result:
        with self._lock:
            if self._result is None:
                raise ValueError
            return self._result

    @property
    def status(self) -> Status:
        with self._lock:
            return self._status

    @property
    def index(self) -> Index:
        return self._index

    @property
    def num_attempts(self) -> int:
        with self._lock:
            return self._num_attempts


class TaskManager:
    def __init__(self, num_tasks: int) -> None:
        self._num_tasks = num_tasks
        self._lock = threading.Lock()
        self._tasks: List[Task] = []
        self._num_occupied_workers = 0
        self._results: Dict[Index, Result] = {}

    @property
    def num_occupied_workers(self) -> int:
        with self._lock:
            return len(self._tasks)

    def add_task(self, task: Task) -> None:
        threading.Thread(target=task.run).start()
        with self._lock:
            self._tasks.append(task)

    def check_tasks_and_record_results(self) -> List[Task]:
        with self._lock:
            in_progress_tasks = []
            failed_tasks = []
            for task in self._tasks:
                if task.status is Status.DONE:
                    self._results[task.index] = task.result
                elif task.status is Status.IN_PROGRESS:
                    in_progress_tasks.append(task)
                elif task.status is Status.FAILED:
                    failed_tasks.append(task)
                else:
                    continue
            self._tasks = in_progress_tasks
            return failed_tasks

    @property
    def done(self) -> bool:
        return len(self._results) == self._num_tasks

    @property
    def results(self) -> List[Result]:
        if not self.done:
            raise ValueError
        results_as_list = [0.0] * self._num_tasks
        for task_index, result in self._results.items():
            results_as_list[task_index] = result
        return results_as_list


def run_tasks(
    tasks: List[Callable[[], Any]],
    max_num_concurrent_workers: int = 5,
    max_num_attempts: int = 5,
    timeout_per_loop: float = 0.1,
    timeout_on_failure_in_seconds: float = 1.0,
) -> List[Result]:
    task_queue: PriorityQueue[Tuple[Priority, Index, Union[Task, EndOfQueue]]] = PriorityQueue()
    unique_index_generator = (
        _generate_unique_index()
    )  # prevents tasks from being compared in the priority queue
    initial_priority = max_num_attempts
    for index, task_ in enumerate(tasks):
        task_queue.put((initial_priority, index, Task(f=task_, index=index)))
    task_queue.put((float("inf"), next(unique_index_generator), EndOfQueue()))
    task_manager = TaskManager(num_tasks=len(tasks))
    last_failed_time: Optional[Time] = None
    while not task_manager.done:
        time.sleep(timeout_per_loop)
        on_timeout = (
            last_failed_time and time.time() - last_failed_time < timeout_on_failure_in_seconds
        )
        if on_timeout:
            print("On timeout")
            continue
        last_failed_time = None
        worker_is_available = task_manager.num_occupied_workers < max_num_concurrent_workers
        if worker_is_available and not task_queue.empty():
            num_attempts_remaining, _, task = task_queue.get()
            if isinstance(task, EndOfQueue):
                continue
            if num_attempts_remaining == 0:
                raise RuntimeError("Task failed more than the maximum number of attempts")
            task_manager.add_task(task)
        failed_tasks = task_manager.check_tasks_and_record_results()
        if failed_tasks:
            last_failed_time = time.time()
        for task in failed_tasks:
            num_attempts_remaining = max_num_attempts - task.num_attempts
            task_queue.put((num_attempts_remaining, next(unique_index_generator), task))
    return task_manager.results


def _generate_unique_index() -> Generator[Index, None, None]:
    index = 0
    while True:
        yield index
        index += 1


if __name__ == "__main__":
    tasks = [random.random for index in range(100)]
    results = run_tasks(tasks, max_num_attempts=5)
    print(results)
