import argparse
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import random
import sympy as sy
import sys
from utils.utils import *

random.seed(student_id)

# set the number of iterations and learning rate
iters = random.randint(100,300)
learning_rate = 0.01

def plot_grad_change(X,Y,Z, c, grad_xs0, grad_xs1, grad_ys):
    fig = plt.figure()
    title_str = student_id_orig+":Gradient Descent:"+"lr="+str(learning_rate)
    plt.title(title_str)
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r,alpha=0.7)
    for i in range(len(grad_xs0)):
        ax.plot([grad_xs0[i]],[grad_xs1[i]], grad_ys[i][0], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7)
    ax.text(grad_xs0[-1],grad_xs1[-1],grad_ys[-1][0][0],
                 "("+str(round(grad_xs0[-1],2))+","+
                     str(round(grad_xs1[-1],2))+"),"+
                     str(round(grad_ys[-1][0][0],2)))
    plt.show()

def GD(start, x, y, z, c, dc, iters, eta):
    px = start.astype(float)
    py = c(px).astype(float)
    print("GD Start Point:", px, py)
    print("Num steps:",iters)
    grad_xs0, grad_xs1, grad_ys = [px[0][0]], [px[1][0]], [py]
    
    for iter in range(iters):
        # 2. Update px using gradient descent
        gradient = dc(px)
        px = px - eta * gradient
        
        # 3. Update py
        py = c(px)
        
        grad_xs0.append(px[0][0])
        grad_xs1.append(px[1][0])
        grad_ys.append(py)
    print("Converged Point:", px, py)
    plot_grad_change(x, y, z, c, grad_xs0, grad_xs1, grad_ys)

def power_method(matrix, num_iterations):
    # Initialize a random vector
    v = np.random.rand(matrix.shape[1], 1)
    
    for i in range(num_iterations):
        # Multiply the matrix with the vector
        v = np.dot(matrix, v)
        
        # Normalize the vector
        v = v / np.linalg.norm(v)
    
        # Calculate the eigenvalue
        eigenvalue = np.dot(np.dot(matrix, v).T, v)
        
        # Print the eigenvalue
        print("Iteration ", i+1, ": Eigenvalue:", eigenvalue[0][0])
    
    # Calculate the final eigenvalue
    eigenvalue = np.dot(np.dot(matrix, v).T, v)
    
    # Return the final eigenvalue and eigenvector
    return eigenvalue[0][0], v

def power_method_new(matrix, num_iterations):
    # Initialize a random vector
    v = np.random.rand(matrix.shape[1], 1)
    
    # Lists to store the normalized and unnormalized eigenvectors
    normalized_eigenvectors = []
    unnormalized_eigenvectors = []
    
    for i in range(num_iterations):
        # Multiply the matrix with the vector
        v = np.dot(matrix, v)
        
        # Save the unnormalized vector
        unnormalized_eigenvectors.append(v.copy())
        
        # Normalize the vector
        v = v / np.linalg.norm(v)
        
        # Save the normalized vector
        normalized_eigenvectors.append(v.copy())
    
        # Calculate the eigenvalue
        eigenvalue = np.dot(np.dot(matrix, v).T, v)
        
        # Print the eigenvalue
        print("Iteration ", i+1, ": Eigenvalue:", eigenvalue[0][0])
    
    # Calculate the final eigenvalue
    eigenvalue = np.dot(np.dot(matrix, v).T, v)
    
    # Return the final eigenvalue, normalized eigenvector, and unnormalized eigenvector
    return eigenvalue[0][0], normalized_eigenvectors[-1], unnormalized_eigenvectors[-1]

def C(x):
    """
    C(x) = 2 * x1 ** 2 + x1 * x2 + 20 * x2 ** 2 - 5 * x1 - 3* x2
    """
    return (x.T@np.array([[2,1],[1,20]])@x)-(np.array([5,3]).reshape(2,1).T@x)

def dC(x):
    """
    Gradient of the function is: 2 * A * x - b
    A = [[2, 1], [1, 20]]
    b = [5, 3]
    x = [x1, x2]
    """
    return 2*np.array([[2,1],[1,20]])@x-np.array([5,3]).reshape(2,1)

"""
Function: 3 * x1 ** 2 + 2 * x2 ** 2 + 20 * cos(x1) * cos(x2)
"""
def L(x):
    # print("\tL:")
    return x.T@np.array([[3, 0], [0, 2]])@x + 20 * np.cos(np.array([1, 0]).T@x) * np.cos(np.array([0, 1]).T@x)

"""
∇L(x) = [6 * x1 - 20 * sin(x1) * cos(x2), 4 * x2 - 20 * cos(x1) * sin(x2)]
"""
def dL(x):
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    return np.array([6 * x.T @ e1 - 20 * np.sin(x.T @ e1) * np.cos(x.T @ e2),
                     4 * x.T @ e2 - 20 * np.cos(x.T @ e1) * np.sin(x.T @ e2)]).reshape(2, 1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('question', choices=['q1', 'q2', 'q3', 'q4'], help='Specify the question to run')
    return parser.parse_args()

def f(x, y):
    return (10 * x**4 - 20 * x**2 * y + x**2 + 10 * y**2 - 2 * x + 1)

def df_dx(x, y):
    return (40 * x**3 - 40 * x * y + 2 * x - 2)

def df_dy(x, y):
    return (-20 * x**2 + 20 * y)

def armijo_rule(x, y, alpha, beta, c):
    grad_x = df_dx(x, y)
    grad_y = df_dy(x, y)
    while f(x - alpha * grad_x, y - alpha * grad_y) > f(x, y) - c * alpha * (grad_x**2 + grad_y**2) + sys.float_info.epsilon:
        alpha *= beta
    return alpha, grad_x, grad_y

def main():
    np.set_printoptions(suppress=True, precision=5)

    args = parse_arguments()
    if args.question == 'q1':
        """
        i) Generate using code a random integer matrix C of size 4x3 and a matrix
        A1 defined as A1 = C^TC and workout its characteristic equation. Using any
        software package, determine the eigenvalues and eigenvectors.
        Deliverables: The matrices C and A1, the computation of the characteristic equation, 
        the eigenvalues and eigenvectors as obtained from the package.
        """
        mtx_rows = 4
        mtx_cols = 3

        # MC = generate_suitable_matrix(mtx_rows, mtx_cols)
        MC = np.array([[17, 10, 18], [11, 19, 20], [20, 16, 17], [20, 12, 14]])

        print("Matrix C:\n", MC)

        A1 = np.dot(MC.T, MC)

        print("Matrix A1 = C^TC:\n", A1)

        eigenvalues, eigenvectors = np.linalg.eig(A1)
        print("Eigenvalues:", eigenvalues) 
        print("Eigenvectors:") 
        pprint(eigenvectors)

        lamda = sy.symbols('λ')
        ceqn = sy.Matrix(A1).charpoly(lamda).as_expr()
        print("Characteristic Equation for A1:", ceqn)
        print("==========================================================================")

        """
        Write code in python programming language to implement the Power method
        and use it to derive the largest eigenvalue λ1 and corresponding eigenvector
        x1 of A1. Find x1_cap = x1/norm(x1) . Compare the values obtained in i) with these values.
        Deliverables: The handwritten code that implements the Power method, the first 10 iterates
        of eigenvalue generated by the algorithm and the final λ1 and x1_cap and a comment on the comparison
        """
        largest_eigenvalue_A1, eigenvector_A1, uev_A1 = power_method_new(A1, 10)
        print("Largest Eigenvalue λ1:", largest_eigenvalue_A1)
        print("Eigenvector (x1) corresponding to λ1:\n", uev_A1)
        print("Normalized Eigenvector (x1_cap) corresponding to λ1:\n", eigenvector_A1) 
        print("==========================================================================")

        """
        Write code to construct matrix A2 = (A1 - x1_cap x1_cap^T A1).
        Use the Power method code written in ii) and use it to derive the largest eigenvalue
        λ2 and corresponding eigenvector x2 of A2.
        Deliverables: The first 10 iterates of λ2 and x2_cap and a comment on the comparison
        """
        A2 = A1 - np.dot(np.dot(eigenvector_A1, eigenvector_A1.T), A1)
        largest_eigenvalue_A2, eigenvector_A2, uev_A2 = power_method_new(A2, 10)
        print("First 10 iterates give λ2:", largest_eigenvalue_A2)
        print("Eigenvector (x2) corresponding to λ2:\n", uev_A2)
        print("Normalized Eigenvector (x2_cap) corresponding to λ2:\n", eigenvector_A2)
        print("==========================================================================")


        """
        Write a code to construct matrix A3 = (A1 - x1_cap x1_cap^T A1 - x2_cap x2_cap^T A )
        Use the power_method( ) function above and use it to derive the
        largest eigenvalue λ3 and corresponding eigenvector x3 of A3
        Deliverables: The first 10 iterates of λ3 and x3_cap and a comment on the comparison.
        """
        A3 = A1 - np.dot(np.dot(eigenvector_A1, eigenvector_A1.T), A1) - np.dot(np.dot(eigenvector_A2, eigenvector_A2.T), A1)
        largest_eigenvalue_A3, eigenvector_A3, uev_A3 = power_method_new(A3, 10)
        print("First 10 iterates give λ3:", largest_eigenvalue_A3)
        print("Eigenvector (x3) corresponding to λ3:\n", uev_A3)
        print("Normalized Eigenvector x3_cap:\n", eigenvector_A3)
        print("==========================================================================")
    elif args.question == 'q2':
        # Initial point
        x0 = 1.15
        y0 = 1.15

        # Parameters for Armijo's Rule
        alpha = 1
        beta = 0.5
        c = 0.25

        print(f"Initial point: Initial step size = {alpha}\tx0 = {x0}\ty0 = {y0}\tf(x0, y0) = {f(x0, y0)}")
        for i in range(2000000):
            optimal_alpha, grad_x, grad_y = armijo_rule(x0, y0, alpha, beta, c)
            x0 -= optimal_alpha * grad_x
            y0 -= optimal_alpha * grad_y
            objective_value = f(x0, y0)

            if objective_value <= sys.float_info.epsilon:
                break
            
            print(f"Iteration {i+1}: Optimal step size = {optimal_alpha}\tx = {x0}\ty = {y0}\tf(x, y) = {objective_value}")
    elif args.question == 'q3':
        lo = -10
        hi = 10
        x1 = round(random.uniform(lo,0),4)
        x2 = round(random.uniform(lo,0),4)
        x = np.linspace(lo, 1, hi)
        y = np.linspace(lo, 1, hi)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i][j] = C(np.array([X[i][j], Y[i][j]]).reshape(2,1))
        # start Gradient Descent
        GD(np.array([x1,x2]).reshape(2,1),X,Y,Z, C, dC, iters, learning_rate)




    elif args.question == 'q4':




        lo = -10
        hi = 10
        x1 = round(random.uniform(lo,0),4)
        x2 = round(random.uniform(lo,0),4)
        x = np.linspace(lo, 1, hi)
        y = np.linspace(lo, 1, hi)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i][j] = L(np.array([X[i][j],Y[i][j]]).reshape(2,1))
        # start Gradient Descent
        GD(np.array([x1,x2]).reshape(2,1),X,Y,Z, L, dL, iters, learning_rate)

if __name__ == "__main__":
    main()

















"""
Matrix C:  [[42 15 36 45]
 [15  9 15 18]
 [36 15 35 36]
 [45 18 36 54]]
Matrix A1 = C^TC:  [[5310 2115 4617 5886]
 [2115  855 1848 2349]
 [4617 1848 4042 5094]
 [5886 2349 5094 6561]]
Eigenvalues: [16701.54839    -0.         11.02779    55.42382]
Eigenvectors:
array([[ 0.56368, -0.63689, -0.52169, -0.06681],
       [ 0.22512, -0.47767,  0.84125, -0.11598],
       [ 0.49016,  0.47767,  0.03969, -0.72801],
       [ 0.62556,  0.37152,  0.13625,  0.67237]])
Characteristic Equation: [       -0. -10208025.   1110456.    -16768.         1.]
Largest Eigenvalue: 16701.54838893725
Corresponding Eigenvector:
array([[0.56368],
       [0.22512],
       [0.49016],
       [0.62556]])
First 10 iterates give λ2: 55.42381986977779
Corresponding eigenvector x2_cap:
array([[-0.06681],
       [-0.11598],
       [-0.72801],
       [ 0.67237]])
First 10 iterates give λ3: 11.027791192972304
Corresponding Eigenvector of A3:
array([[-0.52169],
       [ 0.84125],
       [ 0.03969],
       [ 0.13625]])
"""

"""
+- [Matrix Methods in Data Analysis, Signal Processing, and](https://www.youtube.com/playlist?list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k)
+- [Essential Mathematics for Machine Learning](https://www.youtube.com/watch?v=JO9jNe6BemE&list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE)
+- [Linear Algebra](https://www.youtube.com/watch?v=7UJ4CFRGd-U&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
+- [Machine Learning](https://www.youtube.com/watch?v=KzH1ovd4Ots&list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh)
+- [Machine Learning - Andrew Ng](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
+- [Matrix Calculus For Machine Learning And Beyond](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)
+- [Machine Learning - Stanford University](https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599)
+- [Positive Definite and Semidefinite Matrices](https://www.youtube.com/watch?v=xsP-S7yKaRA)
+- [Applied Linear Algebra](https://www.youtube.com/watch?v=YiCAeeBEo50&list=PLoROMvodv4rMz-WbFQtNUsUElIh2cPmN9&index=5)
"""