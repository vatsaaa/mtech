import time
import platform

if platform.system() == "Windows":
    import psutil
elif platform.system() == "Darwin" or platform.system() == "Linux":
    import resource

if platform.system() == "Windows":
    def track_time_and_space(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss 

            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            print(platform.system(), start_memory, end_memory)

            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024  # Convert to kilobytes

            print(f"Execution time: {execution_time} seconds")
            print(f"Memory usage: {memory_usage} KB")

            return result, execution_time, memory_usage

        return wrapper
else:
    def track_time_and_space(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            print(platform.system(), start_memory, end_memory)

            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024  # Convert to kilobytes

            print(f"Execution time: {execution_time} seconds")
            print(f"Memory usage: {memory_usage} KB")

            return result, execution_time, memory_usage
        
        return wrapper

