'''Assignment Question: Use binary search to find an element in a given array'''

from typing import List
from pprint import pprint

str_arr = input("Please give the array to search within: ")

# Split the user given array on space and create an array of integers
arr = list(map(int, str_arr.split(' ')))

def sort(arr: List[int]) -> List[int]:
    for i in range(1, len(arr), 1):
        for j in range(i+1, len(arr), 1):
            if(arr[i] > arr[j]):
                arr[i], arr[j] = arr[j], arr[i]
    
    return arr

arr = sort(arr)

element = int(input("Give the element to search in this array: "))

low = 0
high = len(arr)
mid = (low + high) // 2

