import numpy as np
import scipy.linalg as la

A: np.ndarray = np.array([[5, 2],
                          [2, 4]])
# B: np.ndarray = np.array([[-4, -2], 
#                           [-2, -2]])


def find_cholesky_decomposition(matrix: np.ndarray, mtx_name: str):
    try:
        L = la.cholesky(matrix, lower=True)
        print("Cholesky Decomposition exists for matrix {mtx_name}")
        print("Cholesky Decomposition for matrix {mtx_name} is:", L)
    except la.LinAlgError:
        print("Cholesky Decomposition does not exist.")

find_cholesky_decomposition(A, "A")
# find_cholesky_decomposition(B, "B")