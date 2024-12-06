import numpy as np
from sympy import Matrix
from typing import Tuple

# Define the matrix
A: np.ndarray = np.array([[0, 1], 
                          [2, 0]])

def svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Perform SVD decomposition
    U: np.ndarray
    S: np.ndarray
    V: np.ndarray
    U, S, V = np.linalg.svd(A)

    # Convert U, S, V to SymPy matrices for better display
    U = Matrix(U)
    S = Matrix(np.diag(S))
    V = Matrix(V)

    # Print the results
    print("U:", U)
    print("S:", S)
    print("V:", V)

    return U, S, V

U, S, V = svd(A)

# def reconstruct(U: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
#     # Reconstruct the original matrix
#     A_reconstructed: np.ndarray = U @ S @ V.T  # Transpose V to match dimensions

#     print("Reconstructed matrix:", A_reconstructed)

#     return A_reconstructed

# X: np.ndarray = np.array([[1/np.sqrt(10),  1/np.sqrt(15), 1/np.sqrt(2), -1/np.sqrt(3)],
#      [-2/np.sqrt(10), 3/np.sqrt(15), 0, 0],
#      [2/np.sqrt(10),  2/np.sqrt(15), 0, 1/np.sqrt(3)],
#      [-1/np.sqrt(10), -1/np.sqrt(15), 1/np.sqrt(2), 1/np.sqrt(3)]])

# Y: np.ndarray = np.array([[3, 0, 0],
#      [0, 2, 0],
#      [0, 0, 0],
#      [0, 0, 0]])

# Z: np.ndarray = np.array([[1/np.sqrt(3), -1/np.sqrt(6), -1/np.sqrt(2)],
#      [-1/np.sqrt(3), -2/np.sqrt(6), 0],
#      [1/np.sqrt(3), -1/np.sqrt(6), 1/np.sqrt(2)]])



# P = X @ Y @ Z.T
# print("A:", P)
# print("A size:", P.shape)
# print("A rank:", np.linalg.matrix_rank(P))