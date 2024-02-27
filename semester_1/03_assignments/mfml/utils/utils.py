import numpy as np

student_id_orig = '2023aa05727'
student_id = '2023aa05727'
# remove non-numeric characters from the student id
student_id = ''.join([i for i in student_id if i.isdigit()])

def generate_positive_definite_symmetric_matrix(size: int):
    # Check if the size is a positive integer
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    
    # Create a random symmetric matrix
    A = np.random.randint(1, 7, (size, size))
    A = np.triu(A) + np.triu(A, 1).T
    
    # Ensure the matrix is positive definite
    C = np.dot(A, A.T)
    
    return C

def generate_suitable_matrix(rows: int, cols: int):
    return np.random.randint(10, 21, (rows, cols))
