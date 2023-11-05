'''
Write a NumPy program to find the number of elements 
of an array, length of one array element in bytes and total 
bytes consumed by the elements
'''
import numpy as np
import inquirer

## Ask the user what type of array they want to create
questions = [
  inquirer.List('Array Type',
                message="What type of NP array do you want to create?",
                choices=['int', 'float', 'str'],
            ),
]
answers = inquirer.prompt(questions)

if answers['Array Type'] == 'int':
    three_arrays = np.empty((3, 3), dtype=int)
elif answers['Array Type'] == 'float':
    three_arrays = np.empty((3, 3), dtype=float)
elif answers['Array Type'] == 'str':
    three_arrays = np.empty((3, 3), dtype=str)
else:
    print("Kindly choose from the supported inputs, please try again.")
    exit()

# Dynamically create 3 arrays of 3 elements of type as specified by the user
for i in range(0, 3, 1):
    user_input = input("Enter 3 numbers separated by comma: ")
    try:
        three_arrays[i] = np.array(user_input.split(','), dtype=answers['Array Type'])
    except ValueError:
        print("Invalid input. Please try again.")
        i -= 1
        continue

# Type of three_arrays and type of elements in the array
print("Type of three_arrays:", type(three_arrays))
print("Type of elements in the array:", three_arrays.dtype)

# print the array
print(three_arrays)

# Number of elements in the array
print("Number of elements in the array:", three_arrays.size)

# Length of one array element in bytes
print("Length of one array element in bytes:", three_arrays.itemsize)

# Total bytes consumed by the elements
print("Total bytes consumed by the elements:", three_arrays.nbytes)

# Print the shape of the array
print("Shape of the array: ", three_arrays.shape)

# Print the dimension of the array
print("You created a ", three_arrays.ndim, " dimensional array.")

# Print the length of the array
print("Length of the array: ", len(three_arrays))

# Print the number of rows in the array
print("Number of rows in the array: ", len(three_arrays[0]))

# Transpose the 2 dimensional array
print("Array transposed: \n", three_arrays.transpose())

# Find the set difference
print("Set difference: ", np.setdiff1d(three_arrays[0], three_arrays[1]))

# Print array in reverse order
print("Array printed in reverse order: \n", three_arrays[::-1])