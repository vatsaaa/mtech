from typing import List

def generate_subsets(numbers: List[int], target: int) -> List[List[int]]:
    """
    Generate all subsets of numbers that sum up to at least target.
    """
    def backtrack(start: int, path: List[int], current_sum: int) -> None:
        if current_sum >= target:
            subsets.append(path[:])  # Append a copy of the path
        for i in range(start, len(numbers)):
            path.append(numbers[i])
            # Recursively call backtrack with updated parameters
            backtrack(i + 1, path, current_sum + numbers[i])
            path.pop()  # Backtrack

    subsets: List[List[int]] = []
    # Convert all non-empty string elements in numbers to integers
    numbers = [int(x) for x in numbers if x.strip() != "" and x.strip().isdigit()]
    backtrack(0, [], 0)
    return subsets

numbers = input("Enter a list of numbers separated by space: ")
target = int(input("Enter the target sum: "))

subsets = generate_subsets(numbers, target)

print(f"Subsets of {numbers} that sum up to at least {target}: {subsets}")