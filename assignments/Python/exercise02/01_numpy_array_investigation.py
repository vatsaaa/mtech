'''
Write a NumPy program to find the number of elements 
of an array, length of one array element in bytes and total 
bytes consumed by the elements
'''
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Number of elements in the array
num_elements = arr.size
print("Number of elements in the array:", num_elements)

# Length of one array element in bytes
elem_bytes = arr.itemsize
print("Length of one array element in bytes:", elem_bytes)

# Total bytes consumed by the elements
total_bytes = arr.nbytes
print("Total bytes consumed by the elements:", total_bytes)
