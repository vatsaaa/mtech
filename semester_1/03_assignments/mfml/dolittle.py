import numpy as np
from scipy.linalg import lu

# Define the matrix
A: np.ndarray = np.array([[8, 11], 
                          [-10, 5]])

# Perform LU decomposition
P, L, U = lu(A)

# Print lower triangle matrix (L)
print("Lower triangle matrix (L):")
print(L)

# Print upper triangle matrix (U)
print("Upper triangle matrix (U):")
print(U)