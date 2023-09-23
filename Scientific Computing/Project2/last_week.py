import numpy as np
# this is going to be useful later
def same(a,b):
    return np.isclose(a,b).all()


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
        vk[k] -= -np.sign(vk[k]) * np.linalg.norm(vk)    # fortegnsfejl i bogen?

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
    
    # assert same(np.linalg.qr(A)[1], R[:min(A.shape),:min(A.shape)]), "R is wrong"

    return VR



def backward_substitute(U, y):
    x = np.empty_like(y)
    N = U.shape[0]
    for k in range(N)[::-1]:
        x[k] = (y[k] - np.dot(U[k, (k+1):N], x[(k+1):N])) / U[k,k]
    return x



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
        vv = np.dot(v,v)
        if vv == 0:
            continue
        c -= 2 * v * np.dot(v,c)/vv


    # solve R1x = c1 as described in the book
    c1 = c[:R1.shape[1]]
    x = backward_substitute(R1, c1)
 
    # assert same(x, np.linalg.lstsq(A,b, rcond = None)[0]), "Numpy says solution is wrong"
    return x