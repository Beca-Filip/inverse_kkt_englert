import numpy as np
import matplotlib.pyplot as plt
from qp_wrappers import *

if __name__ == '__main__':
    
    # Initial position
    xi = np.array([-1, 0])
    # Final position
    xf = np.array([1, 0])
    # Intermediate constraint parameters
    cintmd = np.array([1, 1])
    dintmd = np.zeros(1)
    
    # Intermediate displacement constraint parameters
    ey = np.array([0, 1]).reshape((2, 1))
    cdisplacement = np.hstack([-ey.T, ey.T])
    ddisplacement = -np.ones((1, ))

    # Number of samples
    N = 6
    # State dimension
    n = 2
    
    # Finite differences acceleration matrix
    D = np.zeros(((N-2)*n, N*n))

    for i in range(N-2):
        D[2*i:2*(i+1), 2*i:2*i+2] = np.eye(2)
        D[2*i:2*(i+1), 2*i+2:2*i+4] = -2*np.eye(2)
        D[2*i:2*(i+1), 2*i+4:2*i+6] = np.eye(2)

    # Accelerations a = D @ x
    # Cost function quadratic matrix (a.T @ a)
    P = 2 * D.T @ D + 1e-15 * np.eye(D.shape[1])
    # Cost function linear vector
    q = np.zeros((N*n, ))

    # Initial constraint matrix
    Ai = np.zeros((n, N*n))
    Ai[0:2, 0:2] = np.eye(2)
    # Inital constraint vector
    bi = xi

    # Final constraint matrix
    Af = np.zeros((n, N*n))
    Af[-2:, -2:] = np.eye(2)
    # Final constraint vector
    bf = xf

    # Intermediate constraint matrix
    Aintmd = np.zeros((2, N*n))
    Aintmd[0, (N//2-1) * n : (N//2-1) * n + n] = cintmd
    Aintmd[1, N//2 * n : N//2 * n + n] = cintmd
    # Intermediate constraint vector
    bintmd = np.vstack([dintmd, dintmd]).reshape((2, ))

    # Intermediate displacement inequality matrix
    Gdisplacement = np.zeros((1, N*n))
    Gdisplacement[0, (N//2-1)*n : (N//2+1)*n] = cdisplacement
    # Intermediate displacement inequality vector
    hdisplacement = ddisplacement

    # Form total equality constraint matrix
    A = np.vstack([Ai, Af, Aintmd])
    # Form total equality constraint vector
    b = np.hstack([bi, bf, bintmd])
    # Form total inequality constraint matrix
    G = Gdisplacement
    # Form total inequality constraint vector
    h = hdisplacement
     
    # Solve with quadprog
    solau1 = quadprog_solve_qp(P, q, G, h, A, b)

    # Coordinates
    X = solau1[0][0::2]
    Y = solau1[0][1::2]

    # Constaint
    x11 = -1
    x12 = (dintmd - cintmd[0] * x11) / cintmd[1]
    x21 = 1
    x22 = (dintmd - cintmd[0] * x21) / cintmd[1]
    x1 = [x11, x21]
    x2 = [x12, x22]

    # PETIT PLOT
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(X, Y, 'ro', linestyle='-')
    plt.plot(x1, x2, 'm', linewidth=2)
    plt.grid()
    plt.show()

    print('hi')