import numpy as np
import cvxopt
import quadprog

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    # return np.array(sol['x']).reshape((P.shape[1],))
    return sol

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    # return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)


if __name__ == '__main__':
    x_star = np.array([-1, -1])
    Q = np.eye(2)
    # q = np.zeros((2,), dtype='float64')
    q = -np.dot(x_star, Q)
    G = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype='float64')
    h = np.array([[1, 1, 1, 1]], dtype='float64').reshape((4,))
    
    solau1 = quadprog_solve_qp(Q, q, G, h)
    solau2 = cvxopt_solve_qp(Q, q, G, h)
    print('hi')