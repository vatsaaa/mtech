import numpy as np
from pprint import pprint

# define matrices
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
matrix3 = np.array([[19, 20, 21], [22, 23, 24], [25, 26, 27]])

# multiply matrices
result = np.dot(np.dot(matrix1, matrix2), matrix3)

# print result
pprint(result)
