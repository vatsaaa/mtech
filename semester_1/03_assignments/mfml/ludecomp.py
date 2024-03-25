import numpy as np
from scipy.linalg import lu

# Define the matrix
A: np.ndarray = np.array([[8, 11], 
                          [-10, 5]])

# Perform LU decomposition
P, L, U = lu(A)

# Print the results
print("P:", P)
print("L:", L)
print("U:", U)


# L = [[1, 0, 0],
#      [0.5, 1, 0],
#      [0.5, 0,  1]]


