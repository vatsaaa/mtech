import numpy as np

A: np.ndarray = np.array([[3, 1],[12, 4]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
for i in range(len(eigenvalues)):
    print(f"Eigenvector corresponding to eigenvalue {eigenvalues[i]}:")
    print(eigenvectors[:, i])