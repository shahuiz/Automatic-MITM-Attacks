from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os

# AES parameters
NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)

# variable declaration
total_round = 8 # total round
start_round = 4   # start round, start in {0,1,2,...,total_r-1}
match_round = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid

# linear constriants for XOR operations
XOR_A = np.asarray([
    [0, 0, 0, 0, 0, 0, 1],
    [-1, 0, -1, 0, 1, 0, -2],
    [0, 0, 1, 0, -1, 0, 1],
    [0, -1, 0, -1, 0, 1, 0],
    [0, 0, 0, 1, 0, -1, 0],
    [1, 0, 0, 0, -1, 0, 1],
    [0, 1, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 1, 0, -1],
    [0, 0, 0, 0, 0, 1, -1]])
XOR_B = np.asarray([0,1,0,1,0,0,0,0,0])

m = gp.Model('x')

K_b = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='K_b').values()).reshape((total_round+1, NROW, NCOL))
K_r = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='K_r').values()).reshape((total_round+1, NROW, NCOL))

K_ini_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_b').values()).reshape((NROW, NCOL))
K_ini_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_r').values()).reshape((NROW, NCOL))

def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_df: gp.Var):
    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df]
    m.addMConstr(XOR_A, list(enum), '>=', -XOR_B)

def key_expansion(m:gp.Model, key_size:int, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, K_b: np.ndarray, K_r: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
    Nk = key_size // 32
    for ri in range(total_r + 1):
        r = ri
        if ri == total_r:
            r = -1

        # initial state of key expansion, strictly unknown and no consumed df, allow grey cells
        if r == start_r:
            print(ri, 'ini')
            for i in ROW:
                for j in COL:
                    m.addConstr(K_b[r, i, j] + K_r[r, i, j] >= 1)
                    m.addConstr(K_ini_b[i, j] == K_b[r, i, j])
                    m.addConstr(K_ini_r[i, j] == K_r[r, i, j])
                    m.addConstr(key_cost_bwd[r,i,j] == 0)
                    m.addConstr(key_cost_fwd[r,i,j] == 0)
        
        # fwd direction
        if r > start_r:
            print(ri, 'fwd')
            lr = r - 1
            for j in COL:
                if j == 0:  # special treatment for col 0 as in AES key schedule
                    for i in ROW:
                        rot_i = (i+1) % NROW
                        rot_j = NCOL - 1
                        gen_XOR_rule(m, in1_b=K_b[lr,i,j], in1_r=K_r[lr,i,j], in2_b=K_b[lr,rot_i,rot_j], in2_r=K_r[lr,rot_i,rot_j], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                        #gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[lr,rot_i,rot_j], in2_r=K_b[lr,rot_i,rot_j], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                        m.addConstr(key_cost_fwd[r,i,j] == 0)
                else:
                    for i in ROW:
                        gen_XOR_rule(m, in1_b=K_b[lr,i,j], in1_r=K_r[lr,i,j], in2_b=K_b[r,i,j-1], in2_r=K_r[r,i,j-1], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                        #gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[r,i,j-1], in2_r=K_b[r,i,j-1], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                        m.addConstr(key_cost_fwd[r,i,j] == 0)

        # bwd direction, reverse blue and red cell notation for XOR propagation        
        if r < start_r: 
            print(ri, 'bwd') 
            lr = r + 1
            for j in COL:
                if j == 0:  # special treatment for col 0 as in AES key schedule
                    for i in ROW:
                        rot_i = (i+1) % NROW
                        rot_j = NCOL - 1
                        gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[r,rot_i,rot_j], in2_r=K_b[r,rot_i,rot_j], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                        #gen_XOR_rule(m, in1_b=K_b[lr,i,j], in1_r=K_r[lr,i,j], in2_b=K_b[r,rot_i,rot_j], in2_r=K_r[r,rot_i,rot_j], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                        m.addConstr(key_cost_bwd[r,i,j] == 0)
                else:
                    for i in ROW:
                        gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[lr,i,j-1], in2_r=K_b[lr,i,j-1], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                        #gen_XOR_rule(m, in1_b=K_b[lr,i,j], in1_r=K_r[lr,i,j], in2_b=K_b[lr,i,j-1], in2_r=K_r[lr,i,j-1], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                        m.addConstr(key_cost_bwd[r,i,j] == 0)


key_cost_fwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((total_round+1, NROW, NCOL))
key_cost_bwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((total_round+1, NROW, NCOL))

key_expansion(m, 128, 8, 7, K_ini_b, K_ini_r, K_b, K_r, key_cost_fwd, key_cost_bwd)

#m.addConstr(K_ini_b[0,0] == 1)
#m.addConstr(K_ini_r[0,0] == 0)

for j in COL:  
    for i in ROW:
        m.addConstr(K_ini_b[i,j] == 1)
        m.addConstr(K_ini_r[i,j] == 0)

m.addConstr(key_cost_fwd[4,0,2] == 1)
m.optimize()
m.write('./Related Key/testFiles/KEtest.sol')

solFile = open('./Related Key/testFiles/KEtest.sol', 'r')
K_b = np.ndarray(shape=(8+1,4,4))
K_r = np.ndarray(shape=(8+1,4,4))

for lines in solFile:
    l=str(lines)
    if l.startswith('K_b'):
        r = int(l[4])
        i = int(l[6])
        j = int(l[8])
        K_b[r,i,j]= l[11]
    if l.startswith('K_r'):
        r = int(l[4])
        i = int(l[6])
        j = int(l[8])
        K_r[r,i,j] = l[11]

with open('./Related Key/testFiles/KEtest.out','w') as f:
    for r in range(8+1):
        f.write('\nround: '+str(r)+'\n')
        for i in ROW:
            color=''
            for j in COL:
                if K_b[r,i,j] == 0 and K_r[r,i,j]==0:
                    color+='w'
                if K_b[r,i,j] == 1 and K_r[r,i,j]==0:
                    color+='b'
                if K_b[r,i,j] == 0 and K_r[r,i,j]==1:
                    color+='r'
                if K_b[r,i,j] == 1 and K_r[r,i,j]==1:
                    color+='g'
            f.write(color+'\n')
f.close()