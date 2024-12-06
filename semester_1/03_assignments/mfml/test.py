import argparse
import numpy as np
from scipy.linalg import lu

# Define the matrix
A = np.array([[6, -2, 2], [-2, 3, -1], [2, -1, 3]])
B = [[5,-8,1],[0,0,7],[0,0,-2]]

C = np.dot(A.T, A)



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='LU and Cholesky Decomposition')

    # Add the optional arguments
    parser.add_argument('-l', '--lu', action='store_true', help='Perform LU decomposition')
    parser.add_argument('-c', '--cholesky', action='store_true', help='Perform Cholesky decomposition')
    parser.add_argument('-d', '--diagonalize', action='store_true', help='Perform Eigen decomposition and diagonalization')
    parser.add_argument('-q', '--qr', action='store_true', help='Perform QR decomposition')
    parser.add_argument('-s', '--svd', action='store_true', help='Perform single value decomposition')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Perform LU decomposition if the -l option is provided
    if args.lu:
        # Perform LU decomposition
        P, L, U = lu(C)

        # Print the results
        print("P:")
        print(P)
        print("L:")
        print(L)
        print("U:")
        print(U)
    
    # Perform Cholesky decomposition if the -c option is provided
    if args.cholesky:
        # Perform Cholesky decomposition
        L = np.linalg.cholesky(C)

        # Print the result
        print("L:")
        print(L)

    # Perform eigen decomposition and diagonalization if the -d option is provided
    if args.diagonalize:
        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # Perform diagonalization
        D = np.diag(eigenvalues)
        V = eigenvectors

        # Print the results
        print("Eigenvalues:")
        print(eigenvalues)
        print("Eigenvectors:")
        print(eigenvectors)
        print("Diagonalized matrix D:")
        print(D)
        print("Diagonalizing matrix V:")
        print(V)

    # Perform single value decomposition if the -s option is provided
    if args.svd:
        # Perform single value decomposition
        U, S, V = np.linalg.svd(C)

        # Print the results
        print("U:")
        print(U)
        print("S:")
        print(S)
        print("V:")
        print(V)

    # Perform QR decomposition if the -q option is provided
    if args.qr:
        # Perform QR decomposition
        Q, R = np.linalg.qr(C)

        # Print the results
        print("Q:")
        print(Q)
        print("R:")
        print(R)

def check_diagonalizable(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    if np.all(np.isreal(eigenvalues)):
        P = eigenvectors
        P_inv = np.linalg.inv(P)
        diagonalized_A = np.dot(np.dot(P, A), P_inv)
        print("Matrix A is diagonalizable.")
        print("Eigen Values:", eigenvalues)
        return P, diagonalized_A
    else:
        print("Matrix A is not diagonalizable.")
        return None


def find_linearly_independent_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    linearly_independent_eigenvectors = []

    for i in range(len(eigenvectors[0])):
        # Check if the eigenvector is linearly independent
        is_linearly_independent = True
        for j in range(len(linearly_independent_eigenvectors)):
            stacked_vectors = np.column_stack(linearly_independent_eigenvectors + [eigenvectors[:, i]])
            if np.linalg.matrix_rank(stacked_vectors) <= j:
                is_linearly_independent = False
                break

        if is_linearly_independent:
            linearly_independent_eigenvectors.append(eigenvectors[:, i])

    return linearly_independent_eigenvectors, eigenvalues


if __name__ == "__main__":
    np.set_printoptions(suppress=True,precision=4)
    # main()
    # print(np.linalg.eig(C))
    # print(check_diagonalizable(B))
    # liev, ev = find_linearly_independent_eigenvectors([[1, 0, 0, 0], [0, 3, -3, 0], [0, 1, -1, 0], [0, 0, 0, 3]])
    liev, ev = find_linearly_independent_eigenvectors([[2,1,0],[0,2,0],[0,0,3]])
    print(liev, "\n", ev)
    # rank = np.linalg.matrix_rank(A)
    # print("Rank of the matrix:", rank)
    # if rank == 4:
    #     print("The matrix has 4 linearly independent vectors.")
    # else:
    #     print("The matrix does not have 4 linearly independent vectors.")
    
    # print(np.linalg.eig([[2,5,1], [1,7,-1], [1,0,2]]))
    # eval, evec = np.linalg.eig([[2,0,1],[0,2,0],[1,0,2]])
    # print(eval)
    # Find singular value decomposition if the -svd option is provided
    U, S, V = np.linalg.svd([[1,1,1],[1,1,1]])

    # Print the results
    # print("U:")
    # print(U)
    # print("Singular Values:")
    # print(S)
    # print("V:")
    # print(V)

