import numpy as np

A: np.ndarray = np.array([[1,2,3],
                          [4,5,6],
                          [7,8,9]])

rank = np.linalg.matrix_rank(A)
print("Rank of matrix A:", rank)