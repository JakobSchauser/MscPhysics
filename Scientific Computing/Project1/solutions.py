import numpy as np
from numpy.linalg import norm

from watermatrices import Amat, Bmat, yvec

size = Amat.shape[0]

E = np.block([[Amat, Bmat],
              [Bmat, Amat]])

S = np.block([[np.eye(size),           np.zeros((size,size))], 
              [np.zeros((size,size)), -np.eye(size)]])

z = np.hstack([yvec,-yvec])


data = {"E" : E, "S" : S, "z" : z}


maxnorm = lambda M: np.abs(M).sum(axis=1).max()

condition_number = lambda M: maxnorm(M) * maxnorm(np.linalg.inv(M))

b_bound = lambda w, dw: condition_number(data["E"] - w*data["S"])*maxnorm(dw*data["S"])/maxnorm(data["E"]-w*data["S"])

sigfigs = lambda y, dy: -np.log10(np.abs(dy / y))

# this is going to be useful later
def same(a,b):
    return np.isclose(a,b).all()


def lu_factorize(M, should_pivot=True):
    assert M.shape[0] == M.shape[1], "Matrix should be square"

    # helper function for generating identity matrix
    I = lambda: np.eye(M.shape[0]).astype(float)

    # I am very scared of numpy-stuff being mutable :(
    U = M.copy()

    # get ready for generating L (needed when pivoting)
    Ls = np.empty((M.shape[0]-1,*M.shape))
    Ps = np.empty((M.shape[0]-1,*M.shape))
    
    for i in range(M.shape[0]-1):
        # pivot
        P = I()
        if should_pivot:
        # find best row
            best = np.argmax(U[i:,i]) + i # compensate for looking at submatrix

            # swap
            P[[i,best]] = P[[best,i]]

            # apply
            U = P@U

        # save for generating L
        Ps[i] = P

        # eliminate
        coeff = I()

        # find coefficients
        cc = -U[(i+1):,i]/U[i,i]
        coeff[(i+1):,i] = cc

        # apply
        U = coeff@U

        # save for generating L
        coeff[(i+1):,i] = -cc
        Ls[i] = coeff
    
    # generate L
    L = I()
    for ii in range(len(Ps))[::-1]:
        L = Ls[ii]@L
        L = Ps[ii].T@L


    assert same(L@U, M), "Decomposition did not work"

    return U, L


def forward_substitute(L, z):
    y = np.empty_like(z)

    for k in range(L.shape[0]):
        y[k] = z[k] - np.dot(L[k, 0:k], y[0:k])

    return y


def backward_substitute(U, y):
    x = np.empty_like(y)
    N = U.shape[0]
    for k in range(N)[::-1]:
        x[k] = (y[k] - np.dot(U[k, (k+1):N], x[(k+1):N])) / U[k,k]
    return x


def solve(M, z):
    U, L = lu_factorize(M, should_pivot=False)
    y = forward_substitute(L, z)
    x = backward_substitute(U, y)
    assert same(x, np.linalg.solve(M,z)), "Numpy says solution is wrong"
    assert same(M@x, z), "x is not a solution"
    return x

def solve_alpha(omega):
    M = data["E"]-omega*data["S"]

    x = solve(M, data["z"])
    
    alpha = (data["z"]*x).sum()
    return alpha



def householder_QR_slow(A):
    m = A.shape[0]

    # helper function for generating identity matrix
    I = lambda: np.eye(m).astype(float)

    # Initialize Q and R
    Q = I()
    R = A.copy()

    for k in range(min(A.shape)):

        # Need to zero out the lower part of the k'th column to get in the right shape
        vk = np.zeros(m)

        # copy the lower part of the k'th column
        vk[k:] = R[k:, k].copy()

        # subtract "a_k" in the k'th column
        vk[k] -= -np.sign(vk[k]) * norm(vk)  

        # we dont want to divide by zero
        beta = np.dot(vk, vk)
        if beta == 0:
            continue

        # make the transformation matrix
        H = I() - 2 * np.outer(vk, vk) / beta

        # apply it
        R = H @ R
        Q = H @ Q

    # As can be seen on side 124, we are actually constructing Q^T
    Q = Q.T

    # check that we did it right, as asked
    assert same(Q.T @ Q, np.eye(m)), "Q is not orthogonal"
    assert same(Q@R, A), "QR is wrong"
    
    return Q, R


def householder_fast(A):
    # a lot of this is copied from the slow version
    m, n = A.shape
    I = lambda: np.eye(m).astype(float)

    # initialize R and V
    R = A.copy()
    V = np.zeros(( A.shape[0]+1, A.shape[1])).T # transposing for easier indexing later

    for k in range(min(A.shape)):
        # Need to zero out the lower part of the k'th column to get in the right shape
        vk = np.zeros(m)

        # copy the lower part of the k'th column
        vk[k:] = R[k:, k].copy()

        # subtract "a_k" in the k'th column
        vk[k] -= -np.sign(vk[k]) * norm(vk)    # fortegnsfejl i bogen?

        beta = np.dot(vk, vk)
        
        if beta == 0:
            continue

        # make the transformation matrix
        H = I() - 2 * np.outer(vk, vk) / beta
        
        # apply it, but only to R
        R = H@R

        # save vk (this works because of the transposing earlier)
        V[k,1:] = vk
    
    # re-transpose (is that a word?)
    V = V.T

    # make R same shape as V
    R = np.vstack((R, np.zeros_like(R[0])))
    
    # add V and R for the final result
    VR = V + R
    
    assert same(np.linalg.qr(A)[1], R[:min(A.shape),:min(A.shape)]), "R is wrong"

    return VR


def solve_least_squares(A, b, print_upper = False):
    # find V an R
    VR = householder_fast(A.copy())
    R1 = np.triu(VR)[:A.shape[1],:]
    vs = np.tril(VR, -1)[1:,:]
    if print_upper:
        print("R1")
        print(R1)

    # we dont want to modify b
    c = b.copy()


    # We can simply use the vectors (no need for using a matrix)
    for v in vs.T:
        c -= 2 * v * np.dot(v,c)/np.dot(v,v)

    # solve R1x = c1 as described in the book
    c1 = c[:R1.shape[1]]
    x = backward_substitute(R1, c1)
    
    assert same(x, np.linalg.lstsq(A,b, rcond = None)[0]), "Numpy says solution is wrong"
    return x


def solve_polynomial_1(w_table, alpha_table, N):
    vandermonde_1 = lambda ws, N: np.array([[w**(2*i) for i in range(N+1)] for w in ws])

    vm = vandermonde_1(w_table, N)
    
    sol = solve_least_squares(vm, alpha_table)
    if N == 4:
        print("The coefficients for N = 4:")
        print(sol)
    sol = np.dot(vm, sol)

    return sol


def solve_polynomial_2(w_table, alpha_table, N):
    def get_vandermonde_2(ws, N):
        alphas = -np.array([solve_alpha(w) for w in ws])
        vandermonde_2 = np.array([[w**(i) for i in range(1,N+1)] for w in ws])
        vandermonde_2 = np.concatenate((np.ones(vandermonde_2.shape[0])[:,np.newaxis],vandermonde_2, (alphas * vandermonde_2.T).T), axis=1)
        return vandermonde_2

    def get_single_vandermonde(ws, N, start_from = 0):
        return np.array([[w**i for i in range(start_from,N+1)] for w in ws])
    
    vm = get_vandermonde_2(w_table, N)

    sol = solve_least_squares(vm, alpha_table)

    a = sol[:N+1]
    b = sol[N+1:]

    single_vandermonde = get_single_vandermonde(w_table, N)

    sol_a = np.dot(single_vandermonde, a)
    sol_b = np.dot(single_vandermonde[:,1:], b)

    return sol_a/(1 + sol_b)