import numpy as np
import gurobipy as gp
from gurobipy import GRB

# A and B are constriants for XOR operations, derived by SAGEMATH
# all valid i/o vectors must satisfy: Ax + B >= 0
# i/o vectors are formulated as: (in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df)

XOR_biD_k_LHS = np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, -1, 0, -1, 0, 1, 0, -2, 0], [0, -1, -1, 0, 0, 0, 1, -1, -1], [0, 0, 1, 0, -1, 0, 0, 0, 1], [0, 0, 0, 1, 0, -1, 0, 1, 0], [-1, 0, -1, 0, 1, 0, 0, 0, -2], [0, 0, 0, 0, 0, 0, -1, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0, 1], [-1, 0, 0, -1, 0, 0, 1, -1, -1], [0, 1, 0, 0, 0, -1, 0, 1, 0], [0, 0, 1, 1, 0, 0, -1, 0, 0], [1, 0, 1, 0, -1, 1, -1, -1, 1], [0, 1, 0, 1, 1, -1, -1, 1, -1], [1, 1, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1, -1], [0, 0, 0, 0, 1, 0, 0, -1, -1], [0, 0, 0, 0, 0, -1, 1, 0, 0], [0, 0, 0, 0, -1, 0, 1, 0, 0]])
XOR_biD_k_RHS = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

A = XOR_biD_k_LHS
B = XOR_biD_k_RHS

Points = [
    #in1, in2, out, k, cf,cb
    (0,0, 0,0, 0,0, 0, 0, 0),
    (0,0, 0,1, 0,0, 0, 0, 0),
    (0,0, 1,0, 0,0, 0, 0, 0),
    (0,0, 1,1, 0,0, 0, 0, 0),
    
    (0,1, 0,0, 0,0, 0, 0, 0),
    (0,1, 0,1, 0,1, 1, 0, 0), (0,1, 0,1, 1,1, 1, 0, 1),
    (0,1, 1,0, 0,0, 1, 0, 0),
    (0,1, 1,1, 0,1, 1, 0, 0),
    
    (1,0, 0,0, 0,0, 0, 0, 0),
    (1,0, 0,1, 0,0, 1, 0, 0),
    (1,0, 1,0, 1,0, 1, 0, 0), (1,0, 1,0, 1,1, 1, 1, 0),
    (1,0, 1,1, 1,0, 1, 0, 0),
    
    (1,1, 0,0, 0,0, 0, 0, 0),
    (1,1, 0,1, 0,1, 1, 0, 0),
    (1,1, 1,0, 1,0, 1, 0, 0),
    (1,1, 1,1, 1,1, 1, 0, 0)      
]

# the following code validates if the SAGEMATH tool correctly preserved all valid vectors
from itertools import product
x = [i for i in product(range(2), repeat=9)]
y = []
ans = []

for xi in x:
    y.append(np.asarray(xi))

count = 0
for yi in y:
    flag = 1
    for i in range(len(A)):
        mul = sum(np.multiply(A[i], yi)) + B[i]
        if mul < 0:
            flag = 0
            count +=1
            print(yi, 'violated at ineq', i, 'with val:', mul)
            break
    if flag == 1:
        ans.append(yi)


print('sat:', len(ans), 'unsat:', count)
print('If satisfied amount is correct: ', len(ans)==len(Points))


