import numpy as np
from numpy import newaxis, fill_diagonal, sum, sqrt
NA = newaxis

# Experimental values for Argon atoms
EPSILON=0.997; # kJ/mol
SIGMA=  3.401; # Ångstrom


def distance(points):
    # points: (N,3)-array of (x,y,z) coordinates for N points
    # distance(points): returns (N,N)-array of inter-point distances:
    displacement = points[:, NA] - points[NA, :]
    return sqrt(sum(displacement*displacement, axis=-1))


def LJ(sigma=SIGMA, epsilon=EPSILON):
    def V(points):
        # points: (N,3)-array of (x,y,z) coordinates for N points
        dist = distance(points)

        # Fill diagonal with 1, so we don't divide by zero
        fill_diagonal(dist, 1)

        # dimensionless reciprocal distance
        f = sigma/dist

        # calculate the interatomic potentials
        pot = 4*epsilon*(f**12 - f**6)

        # Undo any diagonal terms (the particles don't interact with themselves)
        fill_diagonal(pot, 0)

        # We could return the sum of just the upper triangular part, corresponding
        # to equation 2 in the assignment text. But since the matrix is symmetric,
        # we can just sum over all elements and divide by 2
        return sum(pot)/2
    return V


def LJgradient(sigma=SIGMA, epsilon=EPSILON):
    def gradV(X):
        d = X[:, NA] - X[NA, :]
        r = sqrt(sum(d*d, axis=-1))

        fill_diagonal(r, 1)

        T = 6*(sigma**6)*(r**-7)-12*(sigma**12)*(r**-13)
        # (N,N)−matrix of r−derivatives
        # Using the chain rule , we turn the (N,N)−matrix of r−derivatives into
        # the (N,3)−array of derivatives to Cartesian coordinate: the gradient.
        # (Automatically sets diagonal to (0,0,0) = X[ i]−X[ i ])
        u = d/r[:, :, NA]
        # u is (N,N,3)−array of unit vectors in direction of X[ i ]−X[ j ]
        return 4*epsilon*sum(T[:, :, NA]*u, axis=1)
    return gradV


# wrapper functions to generate "flattened" versions of the potential and gradient.
def flatten_function(f):
    return lambda x: f(x.reshape(-1, 3))


def flatten_gradient(f):
    return lambda x: f(x.reshape(-1, 3)).reshape(-1)

# potential and gradient with values for Argon
V = LJ()
gradV = LJgradient()

# Flattened potential and gradient.
flat_V     = flatten_function(V)
flat_gradV = flatten_gradient(gradV)
