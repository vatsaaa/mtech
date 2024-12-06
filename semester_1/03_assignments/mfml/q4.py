import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np
from scipy.linalg import eigh
from sympy import cos, Matrix, symbols, diff
##### To be Updated #####
# e.g.,if your BITS email id is 023ab12345@wilp.bits-pilani.com 
# update the below line as student_id = "023xx12345"
student_id = "2023aa05727"
#########################
# set the number of iterations and learning rate
iters = random.randint(100,300)
learning_rate = 0.01
student_id = ''.join([i for i in student_id if i.isdigit()])
random.seed(student_id)

def plot_grad_change(X,Y,Z, c, grad_xs0, grad_xs1, grad_ys):
    fig = plt.figure()
    title_str = "Gradient Descent:"+"lr="+str(learning_rate)
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

"""
Function L(x): 3 * x1 ** 2 + 2 * x2 ** 2 + 20 * cos(x1) * cos(x2)
"""
def C(x):
    return x.T@np.array([[3, 0], [0, 2]])@x + 20 * np.cos(np.array([1, 0]).T@x) * np.cos(np.array([0, 1]).T@x)

"""
∇L(x) = [6 * x1 - 20 * sin(x1) * cos(x2), 4 * x2 - 20 * cos(x1) * sin(x2)]
"""
def dC(x):
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    return np.array([6 * x.T @ e1 - 20 * np.sin(x.T @ e1) * np.cos(x.T @ e2),
                     4 * x.T @ e2 - 20 * np.cos(x.T @ e1) * np.sin(x.T @ e2)]).reshape(2, 1)

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
        Z[i][j] = C(np.array([X[i][j],Y[i][j]]).reshape(2,1))
# start Gradient Descent: Function pointers L and dL are passed as arguments
GD(np.array([x1,x2]).reshape(2,1),X,Y,Z, C, dC, iters, learning_rate)







def is_positive_definite(matrix):
    eigenvalues, _ = np.linalg.eigh(matrix)
    return np.all(eigenvalues > 0)

# Check convexity
def check_convexity():
    x1, x2 = symbols('x1 x2')
    
    # Define the symbolic expression for the objective function
    L_expr = 3 * x1**2 + 2 * x2**2 + 20 * cos(x1) * cos(x2)
    
    # Compute the Hessian matrix symbolically
    hessian_matrix_sym = Matrix([[diff(L_expr, x1, x1), diff(L_expr, x1, x2)],
                                 [diff(L_expr, x2, x1), diff(L_expr, x2, x2)]])
    
    # Convert the Hessian matrix to a NumPy array
    hessian_matrix_num = np.array(hessian_matrix_sym.subs({x1: 0, x2: 0}), dtype=float)
    
    # You may need to substitute x1 and x2 with specific values if symbolic expression is used.
    if is_positive_definite(hessian_matrix_num):
        print("The function is convex.")
    else:
        print("The function is not convex.")

# Call the convexity check function
check_convexity()