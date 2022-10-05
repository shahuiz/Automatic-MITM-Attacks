from decimal import ROUND_HALF_DOWN
from logging import raiseExceptions
from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import sys
import math

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

def gen_ENT_SupP_rule(m: gp.Model, X_b, X_r, fX_b, fX_r, bX_b, bX_r):
    # seperate MC states into superposition: MC[b,r] -> MC_fwd[b,r] + MC_bwd[b,r]
    # truth table: (1,0)->(1,0)+(1,1); 
    #              (0,1)->(1,1)+(0,1); 
    #              (1,1)->(1,1)+(1,1); 
    #              (0,0)->(0,0)+(0,0);
    for i in ROW:
        for j in COL: 
            m.addConstr(fX_b[i,j] == gp.or_(X_b[i,j], X_r[i,j]))
            m.addConstr(fX_r[i,j] == X_r[i,j])
            m.addConstr(bX_b[i,j] == X_b[i,j])
            m.addConstr(bX_r[i,j] == gp.or_(X_b[i,j], X_r[i,j]))

# generate rules when the states exit SupP
def gen_EXT_SupP_rule(m: gp.Model, fX_b, fX_r, bX_b, bX_r, X_b, X_r):
    A = np.asarray([[-1, 0, -1, 0, 1, 0],
    [0, 0, 1, 0, -1, 0],
    [0, -1, 0, -1, 0, 1],
    [0, 0, 0, 1, 0, -1],
    [1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])

    B = np.asarray([1,0,1,0,0,0,0,0])
    for i in ROW:
        for j in COL:
            enum = [fX_b[i,j], fX_r[i,j], bX_b[i,j], bX_r[i,j], X_b[i,j], X_r[i,j]]
            m.addMConstr(A, list(enum), '>=', -B)


def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_df: gp.Var):
    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df]
    m.addMConstr(XOR_A, list(enum), '>=', -XOR_B)

def key_expansion(m:gp.Model, key_size:int, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, fKeyS_b, fKeyS_r, bKeyS_b, bKeyS_r, fK_b: np.ndarray, fK_r: np.ndarray, bK_b: np.ndarray, bK_r: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
    m.update()
    # set key parameters
    Nk = key_size // 32
    Nb = 4
    Nr = math.ceil((total_r + 1)*Nb / Nk)

    for r in range(Nr):
        # initial state
        if r == start_r: 
            for j in range(Nk):
                print("start",r,j)
                for i in ROW:
                    m.addConstr(fKeyS_b[r, i, j] == gp.or_(K_ini_b[i, j], K_ini_r[i, j]))
                    m.addConstr(fKeyS_r[r, i, j] == K_ini_r[i, j])
                    m.addConstr(bKeyS_b[r, i, j] == K_ini_b[i, j])
                    m.addConstr(bKeyS_r[r, i, j] == gp.or_(K_ini_b[i, j], K_ini_r[i, j]))

                    m.addConstr(key_cost_bwd[r,i,j] == 0)
                    m.addConstr(key_cost_fwd[r,i,j] == 0)
            continue
        # fwd direction
        elif r > start_r:
            for j in range(Nk):            
                if j == 0:    # rotation
                    pr, pj = r-1, Nk-1
                    fTemp_b, fTemp_r = np.roll(fKeyS_b[pr,:,pj], -1), np.roll(fKeyS_r[pr,:,pj], -1)
                    bTemp_b, bTemp_r = np.roll(bKeyS_b[pr,:,pj], -1), np.roll(bKeyS_r[pr,:,pj], -1)
                    print('after rot:\nfwd\n', fTemp_b,'\n', fTemp_r, 'bwd\n', bTemp_b, bTemp_r)
                else:               
                    pr, pj = r, j-1
                    fTemp_b, fTemp_r = fKeyS_b[pr,:,pj], fKeyS_r[pr,:,pj] 
                    bTemp_b, bTemp_r = bKeyS_b[pr,:,pj], bKeyS_r[pr,:,pj] 
                qr, qj = r-1, j      # compute round and column params for w[i-Nk]
                for i in ROW:
                    # since the state is superpositioned, the XOR rule works backward: need to reverse the cost
                    gen_XOR_rule(m, in1_b=fKeyS_r[qr,i,qj], in1_r=fKeyS_b[qr,i,qj], in2_b=fTemp_r[i], in2_r=fTemp_b[i], out_b=fKeyS_r[r,i,j], out_r=fKeyS_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                    gen_XOR_rule(m, in1_b=bKeyS_b[qr,i,qj], in1_r=bKeyS_r[qr,i,qj], in2_b=bTemp_b[i], in2_r=bTemp_r[i], out_b=bKeyS_b[r,i,j], out_r=bKeyS_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                
                # if the state is outside the range, then force the cost as 0
                if r*Nk+j >= total_r*Nb:
                    for i in ROW:
                        m.addConstr(key_cost_fwd[r,i,j] == 0)
                        m.addConstr(key_cost_bwd[r,i,j] == 0)
                print("fwd", r,j,' from temp:', pr, pj, 'w[i-Nk]:', qr, qj)
            continue
        # bwd direction
        elif r < start_r:  
            for j in range(Nk):
                if j == 0:    # rotation
                    pr, pj = r, Nk-1
                    fTemp_b, fTemp_r = np.roll(fKeyS_b[pr,:,pj], -1), np.roll(fKeyS_r[pr,:,pj], -1)
                    bTemp_b, bTemp_r = np.roll(bKeyS_b[pr,:,pj], -1), np.roll(bKeyS_r[pr,:,pj], -1)
                    print('after rot:\nfwd\n', fTemp_b,'\n', fTemp_r, 'bwd\n', bTemp_b, bTemp_r)
                else:               
                    pr, pj = r+1, j-1
                    fTemp_b, fTemp_r = fKeyS_b[pr,:,pj], fKeyS_r[pr,:,pj] 
                    bTemp_b, bTemp_r = bKeyS_b[pr,:,pj], bKeyS_r[pr,:,pj] 
                qr, qj = r+1, j      # compute round and column params for w[i-Nk]
                for i in ROW:
                    gen_XOR_rule(m, in1_b=fKeyS_r[qr,i,qj], in1_r=fKeyS_b[qr,i,qj], in2_b=fTemp_r[i], in2_r=fTemp_b[i], out_b=fKeyS_r[r,i,j], out_r=fKeyS_b[r,i,j], cost_df= key_cost_fwd[r,i,j])
                    gen_XOR_rule(m, in1_b=bKeyS_b[qr,i,qj], in1_r=bKeyS_r[qr,i,qj], in2_b=bTemp_b[i], in2_r=bTemp_r[i], out_b=bKeyS_b[r,i,j], out_r=bKeyS_r[r,i,j], cost_df= key_cost_bwd[r,i,j])
                print("bwd", r,j, ' from temp:', pr, pj, 'w[i-Nk]:', qr, qj)
        else:
            raise Exception("Irregular Behavior at encryption")
        m.update()

    # Match the value to the key states used in the add key state
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_r):
        for j in COL:
            print(r,j,'in KeyS',KeyS_r,KeyS_j)
            for i in ROW:
                fK_b[r,i,j] = fKeyS_b[KeyS_r,i,KeyS_j]
                fK_r[r,i,j] = fKeyS_r[KeyS_r,i,KeyS_j]
                bK_b[r,i,j] = bKeyS_b[KeyS_r,i,KeyS_j]
                bK_r[r,i,j] = bKeyS_r[KeyS_r,i,KeyS_j]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0

# variable declaration
key_size = 192
total_round = 9 # total round
start_round = 2   # start round, start in {0,1,2,...,total_r-1}
Nk= key_size // 32
Nb = 4
Nr = math.ceil((total_round + 1)*Nb / Nk)

K_ini_b = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_b').values()).reshape((NROW, Nk))
K_ini_r = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_r').values()).reshape((NROW, Nk))
for i in ROW:
        for j in range(Nk):  
            m.addConstr(K_ini_b[i,j] + K_ini_r[i,j] >= 1)    

# define vars storing the key state in key schedule
fKeyS_b = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKeyS_b').values()).reshape((Nr, NROW, Nk))
fKeyS_r = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKeyS_r').values()).reshape((Nr, NROW, Nk))
bKeyS_b = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKeyS_b').values()).reshape((Nr, NROW, Nk))
bKeyS_r = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKeyS_r').values()).reshape((Nr, NROW, Nk))

# define Key states with superposition, fK for KEY in fwd direction, bK for KEY in bwd direction
fK_b = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='fK_b').values()).reshape((total_round + 1, NROW, NCOL))
fK_r = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='fK_r').values()).reshape((total_round + 1, NROW, NCOL))
fK_g = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='fK_g').values()).reshape((total_round + 1, NROW, NCOL))
fK_w = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='fK_w').values()).reshape((total_round + 1, NROW, NCOL))
bK_b = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='bK_b').values()).reshape((total_round + 1, NROW, NCOL))
bK_r = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='bK_r').values()).reshape((total_round + 1, NROW, NCOL))
bK_g = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='bK_g').values()).reshape((total_round + 1, NROW, NCOL))
bK_w = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='bK_w').values()).reshape((total_round + 1, NROW, NCOL))

# seperate the key state into supperposition
for r in range(total_round + 1):
    for i in ROW:
        for j in COL: 
            m.addConstr(fK_g[r,i,j] == gp.and_(fK_b[r,i,j], fK_r[r,i,j]))
            m.addConstr(fK_w[r,i,j] + fK_b[r,i,j] + fK_r[r,i,j] - fK_g[r,i,j] == 1)
            m.addConstr(bK_g[r,i,j] == gp.and_(bK_b[r,i,j], bK_r[r,i,j]))
            m.addConstr(bK_w[r,i,j] + bK_b[r,i,j] + bK_r[r,i,j] - bK_g[r,i,j] == 1)


key_cost_fwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((Nr, NROW, Nk))
key_cost_bwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((Nr, NROW, Nk))

key_expansion(m, key_size, total_round, start_round, K_ini_b, K_ini_r, fKeyS_b, fKeyS_r, bKeyS_b, bKeyS_r, fK_b, fK_r, bK_b, bK_r, key_cost_fwd, key_cost_bwd)

# AES192 example
for i in ROW:
    for j in range(Nk):
        if (i==1 and j==0) or (i==2 and j==4) or (i==2 and j==5):
            m.addConstr(K_ini_b[i,j] == 0)
            m.addConstr(K_ini_r[i,j] == 1)
        elif (i==0 and j==4) or (i==1 and j==4) or (i==3 and j==4) or (i==0 and j==5):
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 0)
        else:
            m.addConstr(K_ini_b[i,j] == 1)
            m.addConstr(K_ini_r[i,j] == 1)

m.optimize()
dir = './RK_SupP/testFiles/'
m.write(dir+'KStest.sol')




# draw

solFile = open(dir+'KStest.sol', 'r')
fK_b = np.ndarray(shape=(9+1,4,4))
fK_r = np.ndarray(shape=(9+1,4,4))
bK_b = np.ndarray(shape=(9+1,4,4))
bK_r = np.ndarray(shape=(9+1,4,4))

fKEYS_b = np.ndarray(shape=(Nr,4,6))
fKEYS_r = np.ndarray(shape=(Nr,4,6))
bKEYS_b = np.ndarray(shape=(Nr,4,6))
bKEYS_r = np.ndarray(shape=(Nr,4,6))
Key_cost_fwd = np.ndarray(shape=(Nr,4,6))
Key_cost_bwd = np.ndarray(shape=(Nr,4,6))

Sol = dict()
for line in solFile:
    if line[0] != '#':
        temp = line
        temp = temp.split()
        Sol[temp[0]] = round(float(temp[1]))

for r in range(10):
    for i in ROW:
        for j in COL:
            continue
            fK_b[r,i,j]=Sol["fK_b[%d,%d,%d]" %(r,i,j)]
            fK_r[r,i,j]=Sol["fK_r[%d,%d,%d]" %(r,i,j)]
            bK_b[r,i,j]=Sol["bK_b[%d,%d,%d]" %(r,i,j)]
            bK_r[r,i,j]=Sol["bK_r[%d,%d,%d]" %(r,i,j)]

KeyS_r = 0
KeyS_j = 0
for r in range(-1, total_round):
        for j in COL:
            print(r,j,'in KeyS',KeyS_r,KeyS_j)
            for i in ROW:
                fK_b[r,i,j] = Sol["fKeyS_b[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                fK_r[r,i,j] = Sol["fKeyS_r[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bK_b[r,i,j] = Sol["bKeyS_b[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bK_r[r,i,j] = Sol["bKeyS_r[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0

for r in range(Nr):
    for i in ROW:
        for j in range(Nk):
            fKEYS_b[r,i,j]=Sol["fKeyS_b[%d,%d,%d]" %(r,i,j)]
            fKEYS_r[r,i,j]=Sol["fKeyS_r[%d,%d,%d]" %(r,i,j)]
            bKEYS_b[r,i,j]=Sol["bKeyS_b[%d,%d,%d]" %(r,i,j)]
            bKEYS_r[r,i,j]=Sol["bKeyS_r[%d,%d,%d]" %(r,i,j)]

            Key_cost_fwd[r,i,j]=Sol["Key_cost_fwd[%d,%d,%d]" %(r,i,j)]
            Key_cost_bwd[r,i,j]=Sol["Key_cost_bwd[%d,%d,%d]" %(r,i,j)]

def color(b,r):
    if b==1 and r==0:
        return 'b'
    if b==0 and r==1:
        return 'r'
    if b==1 and r==1:
        return 'g'
    if b==0 and r==0:
        return 'w'

with open(dir+'KStest.txt','w') as f:
    for r in range(Nr):
        f.write('KEY_SCHEDULE_'+str(r)+'\n')
        line1 = ''
        line2 = ''
        for i in ROW:
            for j in range(Nk):
                line1 += color(fKEYS_b[r,i,j], fKEYS_r[r,i,j])
                line2 += color(bKEYS_b[r,i,j], bKEYS_r[r,i,j])
                if j == Nk-1:
                    line1 += '   ' + color(fKEYS_b[r,(i+1)%NCOL,j], fKEYS_r[r,(i+1)%NCOL,j])
                    line2 += '   ' + color(bKEYS_b[r,(i+1)%NCOL,j], bKEYS_r[r,(i+1)%NCOL,j])
            line1+='\n'
            line2+='\n'
        f.write(line1+'\n'+line2)
        f.write('\n'*2)

        if Key_cost_fwd[r,:,:].any():
            f.write('KeyExp costs fwdDf: ' + '\n' + str(Key_cost_fwd[r,:,:]) + '\n')
        if Key_cost_bwd[r,:,:].any():
            f.write('KeyExp costs bwdDf: ' + '\n' + str(Key_cost_bwd[r,:,:]) + '\n')
        f.write('\n'*2)   
    
    for r in range(-1, total_round):
        f.write('K#'+ str(r) + '\n')
        line1 = ''
        line2 = ''
        for i in ROW:
            for j in COL:
                line1 += color(fK_b[r,i,j], fK_r[r,i,j])
                line2 += color(bK_b[r,i,j], bK_r[r,i,j])
            line1+='\n'
            line2+='\n'
        f.write(line1+'\n'+line2)
        f.write('\n'*2)
f.close()