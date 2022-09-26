from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import sys

# AES parameters
NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)

#match_round = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid

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

def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_df: gp.Var):
    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df]
    m.addMConstr(XOR_A, list(enum), '>=', -XOR_B)

def key_expansion(m:gp.Model, key_size:int, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, K_b: np.ndarray, K_r: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
    # set key parameters
    Nk = key_size // 32
    Nr = total_r
    Nb = 4
    
    # set territory marker bwd and fwd 
    bwd = start_r * Nb
    fwd = start_r * Nb + Nk
    
    ini_j = 0   # keep track of what column has been read from the initial setting
    for w in range(Nb*(Nr+1)):
        # deal with index for the whiten key
        if w >= Nb*Nr:
            r, j = -1, w%4
            wi= w - Nb*(Nr+1) 
        else:
            r, j = w//4, w%4
            wi = w
        # initial state
        if wi >= bwd and wi < fwd: 
            print("start",r,j, 'from ini',ini_j)
            for i in ROW:
                m.addConstr(K_b[r, i, j] + K_r[r, i, j] >= 1)
                m.addConstr(K_ini_b[i, ini_j] == K_b[r, i, j])
                m.addConstr(K_ini_r[i, ini_j] == K_r[r, i, j])
                m.addConstr(key_cost_bwd[r,i,j] == 0)
                m.addConstr(key_cost_fwd[r,i,j] == 0)
            ini_j += 1
        # fwd direction
        elif wi >= fwd:            
            pr, pj = (wi-1)//NCOL, (wi-1)%NCOL        # compute round and column params for temp
            if (wi - fwd)% Nk == 0:    # rotation
                temp_b, temp_r = np.roll(K_b[pr,:,pj],-1), np.roll(K_r[pr,:,pj],-1)
                print('after rot', temp_b,'\n', temp_r)
            else:               
                temp_b, temp_r = K_b[pr,:,pj], K_r[pr,:,pj] 
            
            qr, qj = (wi-Nk)//NCOL, (wi-Nk)%NCOL      # compute round and column params for w[i-Nk]
            for i in ROW:
                gen_XOR_rule(m, in1_b=K_b[qr,i,qj], in1_r=K_r[qr,i,qj], in2_b=temp_b[i], in2_r=temp_r[i], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                m.addConstr(key_cost_fwd[r,i,j] == 0)
            print("fwd", r,j,' from temp:', pr, pj, 'w[i-Nk]:', qr, qj)
            continue
        # bwd direction
        elif wi < bwd:  
            pr, pj = (wi+Nk-1)//NCOL, (wi+Nk-1)%NCOL        # compute round and column params for temp
            if (bwd - wi) % Nk == 0:    # rotation
                temp_b, temp_r = np.roll(K_b[pr,:,pj],-1), np.roll(K_r[pr,:,pj],-1)
                print('after rot', temp_b,'\n', temp_r)
            else:               
                temp_b, temp_r = K_b[pr,:,pj], K_r[pr,:,pj] 
            qr, qj = (wi+Nk)//NCOL, (wi+Nk)%NCOL      # compute round and column params for w[i-Nk]
            for i in ROW:
                gen_XOR_rule(m, in1_b=K_r[qr,i,qj], in1_r=K_b[qr,i,qj], in2_b=temp_r[i], in2_r=temp_b[i], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                m.addConstr(key_cost_bwd[r,i,j] == 0)
            print("bwd", r,j, ' from temp:', pr, pj, 'w[i-Nk]:', qr, qj)
            continue
        else:
            raise Exception("Irregular Behavior at key expantion")
        m.update()

# variable declaration
key_size = 192
total_round = 9 # total round
start_round = 2   # start round, start in {0,1,2,...,total_r-1}
Nk= key_size // 32

K_b = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='K_b').values()).reshape((total_round+1, NROW, NCOL))
K_r = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='K_r').values()).reshape((total_round+1, NROW, NCOL))

K_ini_b = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_b').values()).reshape((NROW, Nk))
K_ini_r = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_r').values()).reshape((NROW, Nk))


key_cost_fwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((total_round+1, NROW, NCOL))
key_cost_bwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((total_round+1, NROW, NCOL))

key_expansion(m, key_size, total_round, start_round, K_ini_b, K_ini_r, K_b, K_r, key_cost_fwd, key_cost_bwd)

for i in ROW:
    for j in range(Nk):
        continue
        if [i,j] == [0,3] or [i,j] == [2,3] or [i,j] == [3,3]:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 0)
        else:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 1)


# AES192 example
for i in ROW:
    for j in range(Nk):
        #continue
        if (i==1 and j==0) or (i==2 and j==4) or (i==2 and j==5):
            m.addConstr(K_ini_b[i,j] == 0)
            m.addConstr(K_ini_r[i,j] == 1)
        elif (i==0 and j==4) or (i==1 and j==4) or (i==3 and j==4) or (i==0 and j==5):
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 0)
        else:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 1)

# AES256 example 1
for i in ROW:
    for j in range(Nk):
        continue
        if j == 3:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 0)
        else:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 1)

# AES256 example 2
for i in ROW:
    for j in range(Nk):
        continue
        if i == 2:
            if j == 1 or j==2 or j==3:
                m.addConstr(K_ini_b[i,j] == 0)
                m.addConstr(K_ini_r[i,j] == 1)
            elif j==5 or j==6:
                m.addConstr(K_ini_b[i,j] == 1)
                m.addConstr(K_ini_r[i,j] == 0)
            else:
                m.addConstr(K_ini_b[i,j] == 1)
                m.addConstr(K_ini_r[i,j] == 1)
        else:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 1)

m.update()
m.optimize()
m.write('./Related Key/testFiles/KEtest.sol')

#sys.stdout = './keyschedule.txt'

solFile = open('./Related Key/testFiles/KEtest.sol', 'r')
K_b = np.ndarray(shape=(9+1,4,4))
K_r = np.ndarray(shape=(9+1,4,4))

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
    for r in range(9+1):
        if r ==total_round:
            f.write('\nround: '+str(-1)+'\n')
        else:
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