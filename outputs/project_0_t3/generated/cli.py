import threading
from queue import Queue

class ThreadPool:
    def __init__(self, num_workers):
        """
        Initializes the thread pool with the specified number of workers.
        
        :param num_workers: The number of worker threads to create.
        """
        self.num_workers = num_workers
        self.queue = Queue()
        self.lock = threading.Lock()

    def submit(self, task):
        """
        Submits a new task to be executed by one of the worker threads.
        
        :param task: The function or object to execute.
        :return: A tuple containing the ID of the submitted task and the ID of the thread that executed it (or None if no thread is available).
        """
        with self.lock:
            while True:
                try:
                    result = self.queue.get(block=False)
                    return id(result), None
                except queue.Empty:
                    break

    def add_task(self, task):
        """
        Adds a new task to the task queue for execution by worker threads.
        
        :param task: The function or object to execute.
        """
        with self.lock:
            if not self.queue.full():
                # Put the task into the queue
                self.queue.put((task, None))
                return None

    def get_task(self):
        """
        Retrieves the next task from the task queue and executes it on a worker thread.
        
        :return: A tuple containing the ID of the completed task and the ID of the thread that executed it (or None if no thread is available).
        """
        with self.lock:
            while True:
                try:
                    id_result, _ = self.queue.get(block=False)
                    # Execute the task on a worker thread
                    result = threading.Thread(target=self.execute_task, args=(id_result,))
                    result.start()
                    return id(result), None
                except queue.Empty:
                    break

    def execute_task(self, task_id):
        """
        Executes a task on one of the worker threads.
        
        :param task_id: The ID of the task to execute.
        """
        # Simulate some work being done by the task
        import time
        print(f"Executing task {task_id}...")
        time.sleep(2)

def main():
    num_workers = 5
    pool = ThreadPool(num_workers)

    def add_task_to_pool(task):
        id_result, _ = pool.add_task(task)
        print(f"Added task to pool: {id_result}")

    # Add some tasks to the pool
    for i in range(10):
        add_task_to_pool(i)

if __name__ == "__main__":
    main()