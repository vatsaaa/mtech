import numpy as np

def generate_psd_matrix(dim: int) -> np.ndarray:
    # Generate a random matrix
    matrix = np.random.rand(dim, dim)
    
    # Make the matrix symmetric
    matrix = (matrix + matrix.T) / 2
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Set negative eigenvalues to zero
    eigenvalues[eigenvalues < 0] = 0
    
    # Reconstruct the positive semi-definite matrix
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return psd_matrix

# Get the dimensions from the user
dim = int(input("Enter the dimensions of the matrix: "))

# Generate the positive semi-definite matrix
psd_matrix = generate_psd_matrix(dim)

print("Positive Semi-Definite Matrix:")
print(psd_matrix)