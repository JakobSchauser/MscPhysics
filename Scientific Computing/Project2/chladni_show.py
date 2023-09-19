# The functions defined in this file, together with the data file 'chladni-basis.npy'
# allow you to look at your solutions to the Chladni plate problem.
#
# It works quite simply by dotting your coefficient vectors into the set of basis
# functions, defined on a 500x500 grid, then showing the result with pyplot.
#
# show_waves(x) shows the actual wave-functions
# show_nodes(x) shows the wavefunction zeros, where the sand gathers
# show_all_wavefunction_nodes(U,lambdas) shows the zeros of all the eigenfunctions defined by the columns of U

from numpy import sum, load, exp, abs, newaxis as na
import matplotlib.pyplot as plt

basis_set = load("chladni_basis.npy")

def vector_to_function(x,basis_set):
    return sum(x[:,na,na]*basis_set[:,:,:],axis=0) 
    
def show_waves(x,basis_set=basis_set):
    fun = vector_to_function(x,basis_set)
    plt.matshow(fun,origin='lower',extent=[-1,1,-1,1])
    plt.show()

def show_nodes(x,basis_set=basis_set):
    fun   = vector_to_function(x,basis_set)
    nodes = exp(-50*fun**2)  
    plt.matshow(nodes,origin='lower',extent=[-1,1,-1,1],cmap='PuBu')
    plt.show()
    

def show_all_wavefunction_nodes(U,lams,basis_set=basis_set):
    idx = abs(lams).argsort()
    lams, U = lams[idx], U[:,idx]

    N = U.shape[0]    
    m,n = 5,3
    fig, axs = plt.subplots(m,n,figsize=(15,25))
    
    for k in range(N):
        (i,j) = (k//n, k%n)
        fun = vector_to_function(U[:,k],basis_set)
        axs[i,j].matshow(exp(-50*fun**2),origin='lower',extent=[-1,1,-1,1],cmap='PuBu')
        axs[i,j].set_xticklabels([])
        axs[i,j].set_yticklabels([])
        axs[i,j].set_title(r"$\lambda = {:.2f}$".format(lams[k]))
    plt.show()
