import numpy as np
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