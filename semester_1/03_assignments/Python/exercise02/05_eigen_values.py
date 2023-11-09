import numpy as np

# create a square array
arr = np.array([[3, 3], [1, 1]])

# compute the eigenvalues and right eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(arr)

# print the results
print("Eigenvalues:")
print(eigenvalues)
print("\nRight Eigenvectors:")
print(eigenvectors)
