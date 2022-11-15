
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
def find_parents(KeyS:np.ndarray, KeySub, r:int,i:int,j:int):
    level = 0               # store the level exploring
    flag = 1                # store if terminate cond is met
    indices = [[r,i,j]]     # store explored indices 
    while flag: 
        # terminates at current level if either the start_r or a subword is reached
        for x in range(2**level-1, 2**(level+1)-1):
            [xr,xi,xj] = indices[x]
            if xr == start_r:
                flag = 0
                continue
            # fwd dir
            if xr > start_r:
                if xj == 0:
                    pnode = [xr-1, xi]
                    flag = 0
                else: 
                    pnode = [xr, xi, xj-1]
                qnode = [xr-1, xi, xj]
            # bwd dir
            if xr < start_r:
                if xj == 0:
                    pnode = [xr, xi]
                    flag = 0
                else: 
                    pnode = [xr+1, xi, xj-1]
                qnode = [xr+1, xi, xj]
            # if reach start round, terminate terversal after this level
            indices += [pnode,qnode]
            if pnode[0] == start_r or qnode[0] == start_r:
                flag = 0
        # update current level, up to a depth 2
        level += 1
        if level >= 2:
            flag = 0
    
    # reduce nodes with even appearance (xor with itself is null)
    parents = []
    if len(indices) == 1:
        level = 0
    for x in range(2**level-1, 2**(level+1)-1):
        if len(indices[x]) == 2:
            [xr,xi] = indices[x]
            xnode = KeySub[xr,xi]
        else: 
            [xr,xi,xj] = indices[x]
            xnode = KeyS[xr,xi,xj]
        count = 0
        print(KeyS[xr,xi,xj])
        for y in range(2**level-1, 2**(level+1)-1):
            if len(indices[y]) == 2:
                [yr,yi] = indices[y]
                ynode = KeySub[yr,yi]
            else: 
                [yr,yi,yj] = indices[y]
                ynode = KeyS[yr,yi,yj]
            if xnode.sameAs(ynode):
                count += 1
        print(count)
        if count % 2 == 1:
            parents+=[xnode]
    
    # return iterable, redundancy removed parent node list
    return list(set(parents))

    

m = gp.Model('x')
y = m.addVar(vtype=GRB.BINARY)
x = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_x').values()).reshape((Nr, NROW, Nk))
subx = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKSub_x').values()).reshape((Nr, NROW))
m.update()
for r in range(Nr):
    for i in range(4):
        for j in range(Nk):
            continue
            x[r,i,j] = [r,i,j]

output = find_parents(x, subx, 5, 0, 3)
print(output)

