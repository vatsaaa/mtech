
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
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

"""Function C(x) = 2 * x1 ** 2 + x1 * x2 + 20 * x2 ** 2 - 5 * x1 - 3* x2"""
def C(x):
    ##### To be Updated #####
    # 1. Compute and return the gradient
    return (x.T@np.array([[2,1],[1,20]])@x)-(np.array([5,3]).reshape(2,1).T@x)
    #########################

"""
∇C(x)=[ 4x1 +2x2 - 5, 2x1 + 40x2 - 3]^T
dC(x): 2 * A * x - b
A = [[2, 1], [1, 20]]
b = [5, 3]
x = [x1, x2]
"""
def dC(x):
    ##### To be Updated #####
    # NOTE: return value of this function  will 
    #   **ALSO** change for Q4 of the assignment
    #########################
    return 2*np.array([[2,1],[1,20]])@x-np.array([5,3]).reshape(2,1)

def GD(start, x, y, z, c, dc, iters, eta):
    ##### To be Updated #####
    # 2. Update px using gradient descent
    px = start.astype(float)
    py = c(px).astype(float)
    #########################
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
        Z[i][j] = C(np.array([X[i][j], Y[i][j]]).reshape(2,1))
# start Gradient Descent
GD(np.array([x1,x2]).reshape(2,1),X,Y,Z, C, dC, iters, learning_rate)
