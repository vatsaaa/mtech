import random
from typing import List, Optional

def choose_and_remove_random_list(list_of_lists: List[List[int]]) -> Optional[List[int]]:
    if not list_of_lists:
        return None
    
    random_index = random.randint(0, len(list_of_lists) - 1)
    random_list = list_of_lists.pop(random_index)
    print("Removed list:", random_list)
    return random_list

list_of_lists = [[1, 2], [1, 2, 4], [2, 4], [4]]

random_list = choose_and_remove_random_list(list_of_lists)
print("Randomly chosen list:", random_list)
print("Remaining list of lists:", list_of_lists)
