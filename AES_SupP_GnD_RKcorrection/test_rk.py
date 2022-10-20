
from io import TextIOWrapper
import gurobipy as gp
from gurobipy import GRB
from string import Template
import numpy as np
import re
import os
import math

# AES parameters
NROW = 4
NCOL = 4
NBYTE = 32
NGRID = NROW * NCOL
NBRANCH = NROW + 1
ROW = range(NROW)
COL = range(NCOL)
TAB = ' ' * 4

Nk = 6
Nb = 4
Nr = 7
start_r = 3

# define function find all parents in the tree
def find_parents(tree:np.ndarray, r:int,i:int,j:int):
    level = 0               # store the level exploring
    flag = 1                # store if terminate cond is met
    indices = [[r,i,j]]     # store explored indices 
    while flag:
        for x in range(2**level-1, 2**(level+1)-1):
            [xr,xi,xj] = indices[x]
            if xr == start_r:
                flag = 0
                continue
            # fwd dir
            if xr > start_r:
                if xj == 0:
                    pr, pi, pj = xr-1, (xi+1)%Nb, Nk-1
                else: 
                    pr, pi, pj = xr, xi, xj-1
                qr, qi, qj = xr-1, xi, xj
            # bwd dir
            if xr < start_r:
                if j == 0:
                    pr, pi, pj = xr, (xi+1)%Nb, Nk-1
                else: 
                    pr, pi, pj = xr+1, xi, xj-1
                qr, qi, qj = xr+1, xi, xj
            # if reach start round, terminate terversal after this level
            indices += [[pr,pi,pj],[qr,qi,qj]]
            if pr == start_r or qr == start_r:
                flag = 0
        # update current level
        level+=1
    
    # reduce nodes with even appearance (xor with itself is null)
    parents = []
    for x in range(2**level-1, 2**(level+1)-1):
        [xr,xi,xj] = indices[x]
        count = 0
        print(tree[xr,xi,xj])
        for y in range(2**level-1, 2**(level+1)-1):
            [yr,yi,yj] = indices[y]
            if tree[xr,xi,xj].sameAs(tree[yr,yi,yj]):
                count += 1
        print (count)
        if count % 2 == 1:
            parents+=[tree[xr,xi,xj]]
    
    # return iterable, redundancy removed parent node list
    return list(set(parents))

    

m = gp.Model('x')
x = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_x').values()).reshape((Nr, NROW, Nk))
m.update()
for r in range(Nr):
    for i in range(4):
        for j in range(Nk):
            continue
            x[r,i,j] = [r,i,j]

output = find_parents(x, 5, 0, 3)
print(output)

