import time, resource
import platform

if platform.system() == "Windows":
    import psutil

def track_time_and_space(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0
        end_memory = 0

        if platform.system() == "Mac" or platform.system() == "Linux":
            start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        elif platform.system() == "Windows":
            start_memory = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)
        
        end_time = time.time()
        if platform.system == "Mac" or platform.system() == "Linux":
            end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        elif platform.system() == "Windows":
            end_memory = psutil.Process().memory_info().rss

        execution_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024  # Convert to kilobytes

        print(f"Execution time: {execution_time} seconds")
        print(f"Memory usage: {memory_usage} KB")

        return result

    return wrapper