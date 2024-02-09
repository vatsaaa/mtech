import time, resource

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
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = func(*args, **kwargs)
        end_time = time.time()
        end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        execution_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024  # Convert to kilobytes

        print(f"Execution time: {execution_time} seconds")
        print(f"Memory usage: {memory_usage} KB")

        return result

    return wrapper