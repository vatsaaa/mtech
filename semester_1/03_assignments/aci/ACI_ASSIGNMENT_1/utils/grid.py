import time
import psutil

grid = [
    ['S', '.', '.', '.', '#', '#', '.', '.'],
    ['.', 'F', 'F', '.', '.', '.', '.', 'F'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['#', '#', '.', 'F', '.', '.', '.', '.'],
    ['.', 'F', '.', '.', '.', '.', '#', '#'],
    ['.', '.', '.', 'F', '.', '.', '.', '.'],
    ['.', '#', '#', '.', '.', 'F', '.', '.'],
    ['.', '.', '.', '.', '.', 'F', '.', 'G']
]

def track_time_and_space(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
        result = func(*args, **kwargs)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # in MB

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory

        print(f"Execution time: {execution_time} seconds")
        print(f"Memory usage: {memory_usage} MB")

        return result

    return wrapper
