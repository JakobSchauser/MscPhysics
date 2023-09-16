import numpy as np
import matplotlib.pyplot as plt
import cython
import time

q = 10
N : cython.int = 20

T = 0.5

n_runs : cython.int = 100000

def Energy(lattice, i : cython.int, j : cython.int):
    return -np.count_nonzero(lattice[i,j] == [lattice[(i + 1)%N, j], lattice[(i - 1)%N, j], lattice[i, (j + 1)%N], lattice[i, (j - 1)%N]])

def Total_Energy(lattice):
    E = (lattice == np.roll(lattice, 1, axis=0)).sum() + (lattice == np.roll(lattice, 1, axis=1)).sum()
    return E

def flip(lattice, i, j):
    lattice[i, j] = np.random.randint(0, q)

def flip_maybe(lattice, i, j):
    before = lattice[i, j]
    E = Energy(lattice, i, j)
    flip(lattice, i, j)
    if not np.random.rand() < np.exp(-2*(Energy(lattice, i, j)- E)/T):
        lattice[i, j] = before

def run(lattice, n_runs : cython.int, N : cython.int):
    k : cython.int

    for k in range(n_runs):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        flip_maybe(lattice, i, j)

        # if k % (n_runs//10) == 0:
        #     print(k, Total_Energy(lattice)/(N*N))


lattice = np.random.randint(0, q, (N, N), dtype=np.int32)


t = time.time() 
run(lattice, n_runs, N)
print(time.time() - t)

plt.imshow(lattice)
plt.show()