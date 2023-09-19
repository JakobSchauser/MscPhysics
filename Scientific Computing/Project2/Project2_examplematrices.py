# You can use the following matrices to test your eigensolver
# implementations.
#
# Change to the format for whichever language you're using
# (don't write a reader function).
# 
# For the report, you don't need to show results for all 6,
# but I suggest showing result for Aj with the largest j
# that works, and the first j (if any) that fails. This will
# show what your implementation can handle and what it can
# not. 

import numpy as np

# A1-A3 should work with any implementation
A1   = np.array([[1,3],[3,1]]);
eigvals1 = [4,-2];

A2   = np.array([[3,1],[1,3]]);
eigvals2 = [4,2];

A3   = np.array([[1,2,3],[4,3.141592653589793,6],[7,8,2.718281828459045]])
eigvals3 = [12.298958390970709, -4.4805737703355,  -0.9585101385863923];

# A4-A5 require the method to be robust for singular matrices 
A4   = np.array([[1,2,3],[4,5,6],[7,8,9]]);
eigvals4 = [16.1168439698070429897759172023, -1.11684396980704298977591720233, 0]


A5   = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]);
eigvals5 = [68.6420807370024007587203237318, -3.64208073700240075872032373182, 0, 0, 0];

# A6 has eigenvalue with multiplicity and is singular
A6  = np.array(
    [[1.962138439537238,0.03219117137713706,0.083862817159563,-0.155700691654753,0.0707033370776169],
       [0.03219117137713706, 0.8407278248542023, 0.689810816078236, 0.23401692081963357, -0.6655765501236198],
       [0.0838628171595628, 0.689810816078236,   1.3024568091833602, 0.2765334214968566, 0.25051808693319155], 
       [-0.1557006916547532, 0.23401692081963357, 0.2765334214968566, 1.3505754332321778, 0.3451234157557794],
       [0.07070333707761689, -0.6655765501236198, 0.25051808693319155, 0.3451234157557794, 1.5441014931930226]]);
eigvals6 = [2,2,2,1,0]


