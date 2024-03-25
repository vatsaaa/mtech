import numpy as np

def generate_positive_definite_matrix(dim: int) -> np.ndarray:
    # Generate a random matrix
    A = np.random.rand(dim, dim)
    
    # Make the matrix symmetric
    A = (A + A.T) / 2
    
    # Add a small positive constant to the diagonal elements
    A += np.eye(dim) * 0.1
    
    # Check if the matrix is positive definite
    is_positive_definite = np.all(np.linalg.eigvals(A) > 0)
    
    if is_positive_definite:
        return A
    else:
        raise ValueError("Generated matrix is not positive definite")

# Get the dimensions from the user
dim = int(input("Enter the dimensions of the matrix: "))

# Generate the positive definite matrix
matrix = generate_positive_definite_matrix(dim)

print("Generated positive definite matrix:")
print(matrix)
