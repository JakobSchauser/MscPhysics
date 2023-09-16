import numpy as np
import matplotlib.pyplot as plt
import cython
import time


cpdef Energy(lattice, i : cython.int, j : cython.int, N : cython.int):
    return -np.count_nonzero(lattice[i,j] == [lattice[(i + 1)%N, j], lattice[(i - 1)%N, j], lattice[i, (j + 1)%N], lattice[i, (j - 1)%N]])

cpdef Total_Energy(lattice):
    E = (lattice == np.roll(lattice, 1, axis=0)).sum() + (lattice == np.roll(lattice, 1, axis=1)).sum()
    return E

cpdef flip(lattice, i: cython.int, j: cython.int):
    lattice[i, j] = np.random.randint(0, 10)

cpdef flip_maybe(lattice, i: cython.int, j: cython.int, N: cython.int):
    before = lattice[i, j]
    E : cython.int = Energy(lattice, i, j, N)
    flip(lattice, i, j)
    if not np.random.rand() < np.exp(-2*(Energy(lattice, i, j, N)- E)/0.5):
        lattice[i, j] = before

cpdef run(n_runs : cython.int = 100000, N: cython.int = 20):
    lattice = np.random.randint(0, 10, (N, N), dtype=np.int32)
    k : cython.int

    t = time.time()

    for k in range(n_runs):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        flip_maybe(lattice, i, j, N)

        # if k % (n_runs//10) == 0:
        #     print(k, Total_Energy(lattice)/(N*N))

    print(time.time() - t)

    plt.imshow(lattice)
    plt.show()

