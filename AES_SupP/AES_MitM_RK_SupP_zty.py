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
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)
TAB = ' ' * 4

# generate rules when the state enters SupP
def gen_ENT_SupP_rule(m: gp.Model, X_b: gp.Var, X_r: gp.Var, fX_b: gp.Var, fX_r: gp.Var, bX_b: gp.Var, bX_r: gp.Var):
    # seperate MC states into superposition: MC[b,r] -> MC_fwd[b,r] + MC_bwd[b,r]
    # truth table: (1,0)->(1,0)+(1,1); 
    #              (0,1)->(1,1)+(0,1); 
    #              (1,1)->(1,1)+(1,1); 
    #              (0,0)->(0,0)+(0,0);

    m.addConstr(fX_b == gp.or_(X_b, X_r))
    m.addConstr(fX_r == X_r)
    m.addConstr(bX_b == X_b)
    m.addConstr(bX_r == gp.or_(X_b, X_r))

# generate rules when the states exit SupP
def gen_EXT_SupP_rule(m: gp.Model, fX_b: gp.Var, fX_r: gp.Var, bX_b: gp.Var, bX_r: gp.Var, X_b: gp.Var, X_r: gp.Var):
    # truth table: (1,0) + (1,1) or (1,0) -> (1,0)
    #              (0,1) + (1,1) or (0,1) -> (0,1)
    #              (1,1) + (1,1) -> (1,1)
    #              otherwise -> (0,0)
    m.addConstr(X_b == gp.and_(fX_b, bX_b))
    m.addConstr(X_r == gp.and_(fX_r, bX_r))

# generate XOR rule for forward computations, if backward, switch the input of blue and red
def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_fwd: gp.Var, cost_bwd: gp.Var):
    # linear constriants for XOR operations, gennerated by Convex Hull method
    XOR_biD_LHS = np.asarray([[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [-1, 0, -1, 0, 1, 0, 0, -2], [0, 0, 1, 0, -1, 0, 0, 1], [0, -1, 0, -1, 0, 1, -2, 0], [0, 0, 0, 1, 0, -1, 1, 0], [1, 0, 0, 0, -1, 0, 0, 1], [0, 1, 0, 0, 0, -1, 1, 0], [0, 0, 0, 0, 1, 0, -1, -1], [0, 0, 0, 0, 0, 1, -1, -1]])
    XOR_biD_RHS = np.asarray([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])

    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_fwd, cost_bwd]
    m.addMConstr(XOR_biD_LHS, list(enum), '>=', -XOR_biD_RHS)

# generate MC rule with SupP versions
def gen_MC_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_col_u: gp.Var, in_col_x: gp.Var, in_col_y: gp.Var ,out_b: np.ndarray, out_r: np.ndarray, fwd: gp.Var, bwd: gp.Var):
    m.addConstr(NROW*in_col_u + gp.quicksum(out_b) <= NROW)
    m.addConstr(gp.quicksum(in_b) + gp.quicksum(out_b) - NBRANCH*in_col_x <= 2*NROW - NBRANCH)
    m.addConstr(gp.quicksum(in_b) + gp.quicksum(out_b) - 2*NROW*in_col_x >= 0)

    m.addConstr(NROW*in_col_u + gp.quicksum(out_r) <= NROW)
    m.addConstr(gp.quicksum(in_r) + gp.quicksum(out_r) - NBRANCH*in_col_y <= 2*NROW - NBRANCH)
    m.addConstr(gp.quicksum(in_r) + gp.quicksum(out_r) - 2*NROW*in_col_y >= 0)

    m.addConstr(gp.quicksum(out_b) - NROW * in_col_x - bwd == 0)
    m.addConstr(gp.quicksum(out_r) - NROW * in_col_y - fwd == 0)
    m.update()

# generate matching rules, for easy calculation of dm
def gen_match_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_g: np.ndarray, out_b: np.ndarray, out_r: np.ndarray, out_g: np.ndarray, meet_signed, meet):
    m.addConstr(meet_signed == 
        gp.quicksum(in_b) + gp.quicksum(in_r) - gp.quicksum(in_g) +
        gp.quicksum(out_b) + gp.quicksum(out_r) - gp.quicksum(out_g) - NROW)
    m.addConstr(meet == gp.max_(meet_signed, 0))
    m.update()

# set objective function
def set_obj(m: gp.Model, ini_enc_b: np.ndarray, ini_enc_r: np.ndarray, ini_enc_g: np.ndarray, ini_key_b: np.ndarray, ini_key_r: np.ndarray, ini_key_g: np.ndarray, cost_fwd: np.ndarray, cost_bwd: np.ndarray, xor_cost_fwd: np.ndarray, xor_cost_bwd: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray, meet: np.ndarray):
    df_b = m.addVar(lb=1, vtype=GRB.INTEGER, name="DF_b")
    df_r = m.addVar(lb=1, vtype=GRB.INTEGER, name="DF_r")
    dm = m.addVar(lb=1, vtype=GRB.INTEGER, name="Match")
    obj = m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

    m.addConstr(df_b == gp.quicksum(ini_enc_b.flatten()) - gp.quicksum(ini_enc_g.flatten()) + gp.quicksum(ini_key_b.flatten()) - gp.quicksum(ini_key_g.flatten()) - gp.quicksum(cost_fwd.flatten()) - gp.quicksum(xor_cost_fwd.flatten()) - gp.quicksum(key_cost_fwd.flatten()))
    m.addConstr(df_r == gp.quicksum(ini_enc_r.flatten()) - gp.quicksum(ini_enc_g.flatten()) + gp.quicksum(ini_key_r.flatten()) - gp.quicksum(ini_key_g.flatten()) - gp.quicksum(cost_bwd.flatten()) - gp.quicksum(xor_cost_bwd.flatten()) - gp.quicksum(key_cost_bwd.flatten()))
    m.addConstr(dm == gp.quicksum(meet.flatten()))
    m.addConstr(obj - df_b <= 0)
    m.addConstr(obj - df_r <= 0)
    m.addConstr(obj - dm <= 0)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

# key expansion function
def key_expansion(m:gp.Model, key_size:int, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, fKeyS_b: np.ndarray, fKeyS_r: np.ndarray, bKeyS_b: np.ndarray, bKeyS_r: np.ndarray, CONST_0: gp.Var, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
    # set key parameters
    Nk = key_size // 32
    Nb = 4
    Nr = math.ceil((total_r + 1)*Nb / Nk)

    Ksub_b = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='Ksub_b').values()).reshape((Nr, NROW))
    Ksub_r = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='Ksub_r').values()).reshape((Nr, NROW))
    fKsub_b = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKsub_b').values()).reshape((Nr, NROW))
    fKsub_r = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKsub_r').values()).reshape((Nr, NROW))
    bKsub_b = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='bKsub_b').values()).reshape((Nr, NROW))
    bKsub_r = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='bKsub_r').values()).reshape((Nr, NROW))

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
                if j == 0:
                    # RotWord
                    pr, pj = r-1, Nk-1
                    fTemp_b, fTemp_r = np.roll(fKeyS_b[pr,:,pj], -1), np.roll(fKeyS_r[pr,:,pj], -1)
                    bTemp_b, bTemp_r = np.roll(bKeyS_b[pr,:,pj], -1), np.roll(bKeyS_r[pr,:,pj], -1)
                    print('after rot:\nfwd\n', fTemp_b,'\n', fTemp_r, 'bwd\n', bTemp_b, bTemp_r)
                    # SubWord
                    for i in ROW:
                        gen_EXT_SupP_rule(m, fTemp_b[i], fTemp_r[i], bTemp_b[i], bTemp_r[i], Ksub_b[r,i], Ksub_r[r,i])
                        gen_ENT_SupP_rule(m, Ksub_b[r,i], Ksub_r[r,i], fKsub_b[r,i], fKsub_r[r,i], bKsub_b[r,i], bKsub_r[r,i])
                    fTemp_b, fTemp_r = fKsub_b[r], fKsub_r[r] 
                    bTemp_b, bTemp_r = bKsub_b[r], bKsub_r[r] 
                    print('after sub:\nfwd\n', fTemp_b,'\n', fTemp_r, 'bwd\n', bTemp_b, bTemp_r)
                else:               
                    pr, pj = r, j-1
                    fTemp_b, fTemp_r = fKeyS_b[pr,:,pj], fKeyS_r[pr,:,pj] 
                    bTemp_b, bTemp_r = bKeyS_b[pr,:,pj], bKeyS_r[pr,:,pj] 
                qr, qj = r-1, j      # compute round and column params for w[i-Nk]
                for i in ROW:
                    gen_XOR_rule(m, fKeyS_b[qr,i,qj], fKeyS_r[qr,i,qj], fTemp_b[i], fTemp_r[i], fKeyS_b[r,i,j], fKeyS_r[r,i,j], key_cost_fwd[r,i,j], CONST_0)
                    gen_XOR_rule(m, bKeyS_b[qr,i,qj], bKeyS_r[qr,i,qj], bTemp_b[i], bTemp_r[i], bKeyS_b[r,i,j], bKeyS_r[r,i,j], CONST_0, key_cost_bwd[r,i,j])
                
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
                if j == 0:
                    # RotWord
                    pr, pj = r, Nk-1
                    fTemp_b, fTemp_r = np.roll(fKeyS_b[pr,:,pj], -1), np.roll(fKeyS_r[pr,:,pj], -1)
                    bTemp_b, bTemp_r = np.roll(bKeyS_b[pr,:,pj], -1), np.roll(bKeyS_r[pr,:,pj], -1)
                    print('after rot:\nfwd\n', fTemp_b,'\n', fTemp_r, 'bwd\n', bTemp_b, bTemp_r)
                    # SubWord
                    for i in ROW:
                        gen_EXT_SupP_rule(m, fTemp_b[i], fTemp_r[i], bTemp_b[i], bTemp_r[i], Ksub_b[r,i], Ksub_r[r,i])
                        gen_ENT_SupP_rule(m, Ksub_b[r,i], Ksub_r[r,i], fKsub_b[r,i], fKsub_r[r,i], bKsub_b[r,i], bKsub_r[r,i])
                    fTemp_b, fTemp_r = fKsub_b[r], fKsub_r[r] 
                    bTemp_b, bTemp_r = bKsub_b[r], bKsub_r[r] 
                    print('after sub:\nfwd\n', fTemp_b,'\n', fTemp_r, 'bwd\n', bTemp_b, bTemp_r)
                else:               
                    pr, pj = r+1, j-1
                    fTemp_b, fTemp_r = fKeyS_b[pr,:,pj], fKeyS_r[pr,:,pj] 
                    bTemp_b, bTemp_r = bKeyS_b[pr,:,pj], bKeyS_r[pr,:,pj] 
                qr, qj = r+1, j      # compute round and column params for w[i-Nk]
                for i in ROW:
                    gen_XOR_rule(m, fKeyS_b[qr,i,qj], fKeyS_r[qr,i,qj], fTemp_b[i], fTemp_r[i], fKeyS_b[r,i,j], fKeyS_r[r,i,j], key_cost_fwd[r,i,j], CONST_0)
                    gen_XOR_rule(m, bKeyS_b[qr,i,qj], bKeyS_r[qr,i,qj], bTemp_b[i], bTemp_r[i], bKeyS_b[r,i,j], bKeyS_r[r,i,j], CONST_0, key_cost_bwd[r,i,j])
                print("bwd", r,j, ' from temp:', pr, pj, 'w[i-Nk]:', qr, qj)
        else:
            raise Exception("Irregular Behavior at encryption")
        m.update()

# write solution (if there is 0/1/multiple solutions)
def writeSol(m: gp.Model, path):
    if m.SolCount > 0:
        if m.getParamInfo(GRB.Param.PoolSearchMode)[2] > 0:
            gv = m.getVars()
            names = m.getAttr('VarName', gv)
            for i in range(m.SolCount):
                print('SOL', i)
                m.params.SolutionNumber = i
                xn = m.getAttr('Xn', gv)
                lines = ["{} {}".format(v1, v2) for v1, v2 in zip(names, xn)]
                with open(path + '{}_{}.sol'.format(m.modelName, i), 'w') as f:
                    f.write("# Solution for model {}\n".format(m.modelName))
                    f.write("# Objective value = {}\n".format(m.PoolObjVal))
                    f.write("\n".join(lines))
        else:
            m.write(path + m.modelName + '.sol')
        return 1
    else:
        return 0

# simple txt display of the solution
def displaySol(m:gp.Model, path):
    def color(b,r):
        if b==1 and r==0:
            return 'b'
        if b==0 and r==1:
            return 'r'
        if b==1 and r==1:
            return 'g'
        if b==0 and r==0:
            return 'w'

    if not os.path.exists(path= path + m.modelName +'_0.sol'):
        print(path + m.modelName +'_0.sol')
        return

    solFile = open(path + m.modelName +'_0.sol', 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
    match = re.match(r'AES(\d+)RK_(\d+)r_ENC_r(\d+)_Meet_r(\d+)_KEY_r([-+]?\d+)', m.modelName)
    print(m.modelName, match)
    key_size, total_round, enc_start, match_round, key_start = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))

    if enc_start < match_round:
        fwd = list(range(enc_start, match_round))
        bwd = list(range(match_round + 1, total_round)) + list(range(0, enc_start))
    else:
        bwd = list(range(match_round + 1, enc_start))
        fwd = list(range(enc_start, total_round)) + list(range(0, match_round))

    Nb = NCOL
    Nk = key_size // NBYTE
    Nr = math.ceil((total_round + 1)*Nb / Nk)
        
    SB_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    SB_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fSB_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fSB_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bSB_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bSB_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    MC_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fMC_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fMC_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bMC_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bMC_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fAK_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fAK_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAK_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAK_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fKEY_b= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fKEY_r= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKEY_b= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKEY_r= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

    fKEYS_b= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    fKEYS_r= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKEYS_b= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKEYS_r= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    
    Key_cost_fwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    Key_cost_bwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    xor_cost_fwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    xor_cost_bwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    mc_cost_fwd = np.ndarray(shape=(total_round, NCOL), dtype=int)
    mc_cost_bwd = np.ndarray(shape=(total_round, NCOL), dtype=int)

    ini_enc_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_g = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_b = np.ndarray(shape=(NROW, Nk), dtype=int)
    ini_key_r = np.ndarray(shape=(NROW, Nk), dtype=int)
    ini_key_g = np.ndarray(shape=(NROW, Nk), dtype=int)
    
    Meet_fwd_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_fwd_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_bwd_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_bwd_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    meet =  np.ndarray(shape=(NCOL), dtype=int)
    meet_s =  np.ndarray(shape=(NCOL), dtype=int)

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                SB_b[r,i,j]=Sol["S_b[%d,%d,%d]" %(r,i,j)]
                SB_r[r,i,j]=Sol["S_r[%d,%d,%d]" %(r,i,j)]
                fSB_b[r,i,j]=Sol["fS_b[%d,%d,%d]" %(r,i,j)]
                fSB_r[r,i,j]=Sol["fS_r[%d,%d,%d]" %(r,i,j)]
                bSB_b[r,i,j]=Sol["bS_b[%d,%d,%d]" %(r,i,j)]
                bSB_r[r,i,j]=Sol["bS_r[%d,%d,%d]" %(r,i,j)]

                fAK_b[r,i,j]=Sol["fA_b[%d,%d,%d]" %(r,i,j)]
                fAK_r[r,i,j]=Sol["fA_r[%d,%d,%d]" %(r,i,j)]
                bAK_b[r,i,j]=Sol["bA_b[%d,%d,%d]" %(r,i,j)]
                bAK_r[r,i,j]=Sol["bA_r[%d,%d,%d]" %(r,i,j)]

                fMC_b[r,i,j]=Sol["fM_b[%d,%d,%d]" %(r,i,j)]
                fMC_r[r,i,j]=Sol["fM_r[%d,%d,%d]" %(r,i,j)]
                bMC_b[r,i,j]=Sol["bM_b[%d,%d,%d]" %(r,i,j)]
                bMC_r[r,i,j]=Sol["bM_r[%d,%d,%d]" %(r,i,j)]

    for ri in range(total_round):
        for i in ROW:
            for j in COL:
                MC_b[ri, i, j] = SB_b[ri, i, (j + i)%NCOL]
                MC_r[ri, i, j] = SB_r[ri, i, (j + i)%NCOL]

    
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_round):
        for j in COL:
            print(r,j,'in KeyS',KeyS_r,KeyS_j)
            for i in ROW:
                fKEY_b[r,i,j] = Sol["fKeyS_b[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                fKEY_r[r,i,j] = Sol["fKeyS_r[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bKEY_b[r,i,j] = Sol["bKeyS_b[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bKEY_r[r,i,j] = Sol["bKeyS_r[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0
    
    for r in range(Nr):
        for i in ROW:
            for j in range(Nk):
                fKEYS_b[r,i,j] = Sol["fKeyS_b[%d,%d,%d]" %(r,i,j)]
                fKEYS_r[r,i,j] = Sol["fKeyS_r[%d,%d,%d]" %(r,i,j)]
                bKEYS_b[r,i,j] = Sol["bKeyS_b[%d,%d,%d]" %(r,i,j)]
                bKEYS_r[r,i,j] = Sol["bKeyS_r[%d,%d,%d]" %(r,i,j)]

                Key_cost_fwd[r,i,j] = Sol["Key_cost_fwd[%d,%d,%d]" %(r,i,j)]
                Key_cost_bwd[r,i,j] = Sol["Key_cost_bwd[%d,%d,%d]" %(r,i,j)]
    
    for r in range(total_round+1):
        for i in ROW:
            for j in COL:
                xor_cost_fwd[r,i,j] = Sol["XOR_Cost_fwd[%d,%d,%d]" %(r,i,j)]
                xor_cost_bwd[r,i,j] = Sol["XOR_Cost_bwd[%d,%d,%d]" %(r,i,j)]
    
    for r in range(total_round):
        for j in COL:
            mc_cost_fwd[r,j] = Sol["MC_Cost_fwd[%d,%d]" %(r,j)]
            mc_cost_bwd[r,j] = Sol["MC_Cost_bwd[%d,%d]" %(r,j)]

    for i in ROW:
        for j in COL:
            ini_enc_b[i,j] = Sol["E_ini_b[%d,%d]" %(i,j)]
            ini_enc_r[i,j] = Sol["E_ini_r[%d,%d]" %(i,j)]
            ini_enc_g[i,j] = Sol["E_ini_g[%d,%d]" %(i,j)]
    
    for i in ROW:
        for j in range(Nk):
            ini_key_b[i,j] = Sol["K_ini_b[%d,%d]" %(i,j)]
            ini_key_r[i,j] = Sol["K_ini_r[%d,%d]" %(i,j)]
            ini_key_g[i,j] = Sol["K_ini_g[%d,%d]" %(i,j)]

    for i in ROW:
        for j in COL:
            Meet_fwd_b[i,j] = Sol["Meet_fwd_b[%d,%d]" %(i,j)]
            Meet_fwd_r[i,j] = Sol["Meet_fwd_r[%d,%d]" %(i,j)]
            Meet_bwd_b[i,j] = Sol["Meet_bwd_b[%d,%d]" %(i,j)]
            Meet_bwd_r[i,j] = Sol["Meet_bwd_r[%d,%d]" %(i,j)]
    
    for j in COL:
        meet[j] = Sol["Meet[%d]" %j]
        meet_s[j] = Sol["Meet_signed[%d]" %j]

    ini_df_enc_b = np.sum(ini_enc_b[:,:]) - np.sum(ini_enc_g[:,:])
    ini_df_enc_r = np.sum(ini_enc_r[:,:]) - np.sum(ini_enc_g[:,:])

    ini_df_key_b = np.sum(ini_key_b[:,:]) - np.sum(ini_key_g[:,:])
    ini_df_key_r = np.sum(ini_key_r[:,:]) - np.sum(ini_key_g[:,:])

    DF_b = Sol["DF_b"]
    DF_r = Sol["DF_r"]
    Match = Sol["Match"]
    Obj = Sol["Obj"]

    f =  open(path + 'Vis_' + m.modelName +'.txt', 'w')
    f.write('Model:\n')
    f.write(TAB+ 'Total: ' + str(total_round) +'\n')
    f.write(TAB+ 'Start at: r' + str(enc_start) +'\n')
    f.write(TAB+ 'Meet at: r' + str(match_round) +'\n')
    f.write(TAB+ 'KEY start at: r' + str(key_start) +'\n')
    f.write('\nInitialization:\n')
    f.write(TAB+'ENC FWD: ' + str(ini_df_enc_b) + '\n' + TAB+ 'ENC BWD: ' + str(ini_df_enc_r) + '\n')
    f.write(TAB+'KEY FWD: ' + str(ini_df_key_b) + '\n' + TAB+ 'KEY BWD: ' + str(ini_df_key_r) + '\n')
    f.write('\nSolution:\n'+TAB+'Obj= min{DF_b=%d, DF_r=%d, Match=%d} = %d' %(DF_b, DF_r, Match, Obj) + '\n')
    f.write('\nVisualization:\n')
    
    for r in range(total_round):
        header = "r%d  " %r 
        
        if r == match_round:
            header+= 'mat -><-'
        elif r in fwd:
            header+= 'fwd --->'
        elif r in bwd:
            header+= 'bwd <---'
        if r == enc_start:
            header+=TAB*2 + 'ENC_start'
        
        f.write(header + '\n')
        nr = (r+1)%total_round
        
        line1 = ''
        line2 = ''
        for i in ROW:
            SB, MC, fMC, bMC, fAK, bAK, fKEY, bKEY, fSB, bSB, SBN, KEY = '','','','','','','','','','','','' 
            for j in COL:
                SB+=color(SB_b[r,i,j], SB_r[r,i,j])
                MC+=color(MC_b[r,i,j], MC_r[r,i,j])
                fMC+=color(fMC_b[r,i,j], fMC_r[r,i,j])
                bMC+=color(bMC_b[r,i,j], bMC_r[r,i,j])
                fAK+=color(fAK_b[r,i,j], fAK_r[r,i,j])
                bAK+=color(bAK_b[r,i,j], bAK_r[r,i,j])
                
                fKEY+=color(fKEY_b[r,i,j], fKEY_r[r,i,j])
                bKEY+=color(bKEY_b[r,i,j], bKEY_r[r,i,j])

                fSB+=color(fSB_b[nr,i,j], fSB_r[nr,i,j])
                bSB+=color(bSB_b[nr,i,j], bSB_r[nr,i,j])
                SBN+=color(SB_b[nr,i,j], SB_r[nr,i,j])
            
            if r == total_round - 1:
                fAK = '////'
                bAK = '////'

            line1 += SB + TAB*2 + MC + TAB*2+ fMC + TAB*2 + fAK + TAB*2 + fKEY + TAB*2 + fSB + TAB*2 + SBN + '\n'
            line2 += TAB+ TAB*2 + TAB+ TAB*2+ bMC + TAB*2 + bAK + TAB*2 + bKEY + TAB*2 + bSB + '\n' 
        
        f.write('SB#%d'%r +TAB*2+'MC#%d' %r +TAB*2+'fMC#%d   ' %r + TAB +'fAK#%d   '%r +TAB+'fKEY#%d  '%r +TAB+ 'fSB#%d   '%nr +TAB+ 'SB#%d'%nr+ '\n')
        f.write(line1 + '\n')
        f.write(TAB*3           +TAB*3            +'bMC#%d   ' %r + TAB +'bAK#%d   '%r +TAB+'bKEY#%d  '%r +TAB+ 'bSB#%d   '%nr + '\n')
        f.write(line2 + '\n')
        
        if mc_cost_fwd[r,:].any() or mc_cost_bwd[r,:].any():
            f.write('MixCol costs fwdDf: '+ str(mc_cost_fwd[r,:]) + TAB+ 'bwdDf: ' +str(mc_cost_bwd[r,:])+ '\n')
        if xor_cost_fwd[r,:,:].any():
            f.write('AddKey costs fwdDf: ' + '\n' + str(xor_cost_fwd[r,:,:]) + '\n')
        if xor_cost_bwd[r,:,:].any():
                f.write('AddKey costs bwdDf: ' + '\n' + str(xor_cost_bwd[r,:,:]) + '\n')
        f.write('\n')
        
        if r == match_round and match_round != total_round - 1:
            f.write('Match Thru MC:'+'\n'+ 'MC#%d' %r +TAB*2+ 'Meet_BWD' +'\n')
            for i in ROW:
                MC = ''
                Meet_B = ''
                for j in COL:
                    MC+=color(MC_b[r,i,j], MC_r[r,i,j])
                    Meet_B+=color(Meet_bwd_b[i,j], Meet_bwd_r[i,j])
                f.write(MC+TAB*2+Meet_B+'\n') 
            f.write('Degree of Matching:' + str(meet[:]) + '\n'*2)

    # process whiten key
    r = -1
    nr = 0
    line1 = ''
    line2 = ''
    for i in ROW:
        fKEY, bKEY, fAT, bAT, fNSB, bNSB, NSB = '', '', '', '', '', '', ''
        for j in COL:
            fAT +=color(fAK_b[r,i,j], fAK_r[r,i,j])
            bAT +=color(bAK_b[r,i,j], bAK_r[r,i,j])
            fKEY +=color(fKEY_b[r,i,j], fKEY_r[r,i,j])
            bKEY +=color(bKEY_b[r,i,j], bKEY_r[r,i,j])
            fNSB += color(fSB_b[0,i,j], fSB_r[0,i,j])
            bNSB += color(bSB_b[0,i,j], bSB_r[0,i,j])
            NSB += color(SB_b[nr,i,j], SB_r[nr,i,j])
        if match_round == total_round -1:
                NSB = '////'
        line1 += 9*TAB + fAT+ TAB*2 + fKEY + '\n'
        line2 += 9*TAB + bAT+ TAB*2 + bKEY + '\n'
    f.write(TAB*9 +'fAT     ' +TAB+'fKEY#%d  '%r +TAB+ 'fSB#%d   '%nr +TAB+ 'SB#%d' %nr + '\n')
    f.write(line1 + '\n')
    f.write(TAB*9 +'bAT     ' +TAB+'bKEY#%d  '%r +TAB+ 'bSB#%d   '%nr + '\n')
    f.write(line2 + '\n')
    
    tr = total_round
    if xor_cost_fwd[tr,:,:].any():
            f.write('AddKey costs fwdDf: ' + '\n' + str(xor_cost_fwd[tr,:,:]) + '\n')
    if xor_cost_bwd[tr,:,:].any():
            f.write('AddKey costs bwdDf: ' + '\n' + str(xor_cost_bwd[tr,:,:]) + '\n')
    
    if match_round == total_round - 1:
        f.write('\n' + "Identity Match:" + '\n')
        f.write('Meet_FWD'+ TAB + 'SB#0' + '\n')
        for i in ROW:
            SB = ''
            MF = ''
            for j in COL:
                SB +=color(SB_b[0,i,j], SB_r[0,i,j])
                MF +=color(Meet_fwd_b[i,j], Meet_fwd_r[i,j])
            f.write(MF+ TAB*2 + SB + '\n')

    f.write('\n'+'Key Schedule: starts at r'+str(key_start)+'\n')
    
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

    f.close()
    return 'Obj= min{DF_b=%d, DF_r=%d, Match=%d} = %d' %(DF_b, DF_r, Match, Obj)
####################################################################################################################

# interable solve function with parameters
def solve(key_size:int, total_round:int, start_round:int, match_round:int, key_start_round:int, dir):
    # define optimization model
    m = gp.Model('AES%dRK_%dr_ENC_r%d_Meet_r%d_KEY_r%d' % (key_size, total_round, start_round, match_round, key_start_round))
    
    # Calculate Nk for key schedule
    Nb = NCOL
    Nk = key_size // NBYTE
    Nr = math.ceil((total_round + 1)*Nb / Nk)
    
    # assign forward and backward rounds, excluding match round and last round
    if start_round < match_round:
        fwd = list(range(start_round, match_round))
        bwd = list(range(match_round+1, total_round)) + list(range(0, start_round))
    else:
        bwd = list(range(match_round+1, start_round))
        fwd = list(range(start_round, total_round)) + list(range(0, match_round))

#### Define Variables ####
    # define a constant zero, to force cost of df as 0
    CONST0 = m.addVar(vtype = GRB.BINARY, name='Const0')
    m.addConstr(CONST0 == 0)

    # define vars to track the start state of Encryption states (E) 
    E_ini_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='E_ini_b').values()).reshape((NROW, NCOL))
    E_ini_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='E_ini_r').values()).reshape((NROW, NCOL))
    E_ini_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='E_ini_g').values()).reshape((NROW, NCOL)) 
    # add constriants for grey indicators
    for i in ROW:
        for j in COL:
            m.addConstr(E_ini_b[i,j] + E_ini_r[i,j] >= 1)
            m.addConstr(E_ini_g[i,j] == gp.and_(E_ini_b[i,j], E_ini_r[i,j]))

    # define vars to track the start state of Key states (K)  
    K_ini_b = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_b').values()).reshape((NROW, Nk))
    K_ini_r = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_r').values()).reshape((NROW, Nk))
    K_ini_g = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_g').values()).reshape((NROW, Nk))
    # add constraints for grey indicators
    for i in ROW:
        for j in range(Nk):  
            m.addConstr(K_ini_b[i,j] + K_ini_r[i,j] >= 1)      
            m.addConstr(K_ini_g[i,j] == gp.and_(K_ini_b[i,j], K_ini_r[i,j]))

    # define vars storing the SB state at each round with encoding scheme
    S_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_b').values()).reshape((total_round, NROW, NCOL))
    S_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_r').values()).reshape((total_round, NROW, NCOL))
    S_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_g').values()).reshape((total_round, NROW, NCOL))
    S_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_w').values()).reshape((total_round, NROW, NCOL))
    # add constraints for grey and white indicators
    for r in range(total_round):
        for i in ROW:
            for j in COL:
                m.addConstr(S_g[r,i,j] == gp.and_(S_b[r,i,j], S_r[r,i,j]))
                m.addConstr(S_w[r,i,j] + S_b[r,i,j] + S_r[r,i,j] - S_g[r,i,j] == 1)

    # create alias storing the MC state at each round with encoding scheme
    M_b = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    M_r = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    M_g = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    M_w = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    # match the cells with alias through shift rows
    for r in range(total_round):
        for i in ROW:
            for j in COL:   
                M_b[r,i,j] = S_b[r,i,(j+i)%NCOL]
                M_r[r,i,j] = S_r[r,i,(j+i)%NCOL]
                M_g[r,i,j] = S_g[r,i,(j+i)%NCOL]
                M_w[r,i,j] = S_w[r,i,(j+i)%NCOL]

    # define MC states with superposition, fM for MC in fwd direction, bM for MC in bwd direction
    fM_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_b').values()).reshape((total_round, NROW, NCOL))
    fM_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_r').values()).reshape((total_round, NROW, NCOL))
    fM_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_g').values()).reshape((total_round, NROW, NCOL))
    fM_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_w').values()).reshape((total_round, NROW, NCOL))
    bM_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_b').values()).reshape((total_round, NROW, NCOL))
    bM_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_r').values()).reshape((total_round, NROW, NCOL))
    bM_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_g').values()).reshape((total_round, NROW, NCOL))
    bM_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_w').values()).reshape((total_round, NROW, NCOL))

    # define vars storing the Add key state with superposition at each round with encoding scheme
    fA_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_b').values()).reshape((total_round, NROW, NCOL))
    fA_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_r').values()).reshape((total_round, NROW, NCOL))
    fA_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_g').values()).reshape((total_round, NROW, NCOL))
    fA_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_w').values()).reshape((total_round, NROW, NCOL))
    bA_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_b').values()).reshape((total_round, NROW, NCOL))
    bA_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_r').values()).reshape((total_round, NROW, NCOL))
    bA_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_g').values()).reshape((total_round, NROW, NCOL))
    bA_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_w').values()).reshape((total_round, NROW, NCOL))

    # define vars storing the state after adding the key with superposition at each round with encoding scheme
    fS_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_b').values()).reshape((total_round, NROW, NCOL))
    fS_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_r').values()).reshape((total_round, NROW, NCOL))
    fS_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_g').values()).reshape((total_round, NROW, NCOL))
    fS_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_w').values()).reshape((total_round, NROW, NCOL))
    bS_b = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_b').values()).reshape((total_round, NROW, NCOL))
    bS_r = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_r').values()).reshape((total_round, NROW, NCOL))
    bS_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_g').values()).reshape((total_round, NROW, NCOL))
    bS_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_w').values()).reshape((total_round, NROW, NCOL))

    # bluk add grey and white constriants
    for r in range(total_round):
        for i in ROW:
            for j in COL:             
                m.addConstr(fM_g[r,i,j] == gp.and_(fM_b[r,i,j], fM_r[r,i,j]))
                m.addConstr(fM_w[r,i,j] + fM_b[r,i,j] + fM_r[r,i,j] - fM_g[r,i,j] == 1)
                m.addConstr(bM_g[r,i,j] == gp.and_(bM_b[r,i,j], bM_r[r,i,j]))
                m.addConstr(bM_w[r,i,j] + bM_b[r,i,j] + bM_r[r,i,j] - bM_g[r,i,j] == 1)
                
                m.addConstr(fA_g[r,i,j] == gp.and_(fA_b[r,i,j], fA_r[r,i,j]))
                m.addConstr(fA_w[r,i,j] + fA_b[r,i,j] + fA_r[r,i,j] - fA_g[r,i,j] == 1)
                m.addConstr(bA_g[r,i,j] == gp.and_(bA_b[r,i,j], bA_r[r,i,j]))
                m.addConstr(bA_w[r,i,j] + bA_b[r,i,j] + bA_r[r,i,j] - bA_g[r,i,j] == 1)

                m.addConstr(fS_g[r,i,j] == gp.and_(fS_b[r,i,j], fS_r[r,i,j]))
                m.addConstr(fS_w[r,i,j] + fS_b[r,i,j] + fS_r[r,i,j] - fS_g[r,i,j] == 1)
                m.addConstr(bS_g[r,i,j] == gp.and_(bS_b[r,i,j], bS_r[r,i,j]))
                m.addConstr(bS_w[r,i,j] + bS_b[r,i,j] + bS_r[r,i,j] - bS_g[r,i,j] == 1) 
       
    # define vars storing the key state in key schedule, in total Nr rounds, with shape NROW*Nk
    fKeyS_b = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKeyS_b').values()).reshape((Nr, NROW, Nk))
    fKeyS_r = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKeyS_r').values()).reshape((Nr, NROW, Nk))
    bKeyS_b = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKeyS_b').values()).reshape((Nr, NROW, Nk))
    bKeyS_r = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKeyS_r').values()).reshape((Nr, NROW, Nk))
    # create alias storing the round keys with SupP
    fK_b = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    fK_r = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    bK_b = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    bK_r = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    # match the states in key schedule to round keys alias
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_round):
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

    # define vars for columnwise encoding for MixCol input, including MC(fwd) and AK(bwd)
    fM_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fM_col_u').values()).reshape((total_round, NCOL))
    fM_col_x = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fM_col_x').values()).reshape((total_round, NCOL))
    fM_col_y = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fM_col_y').values()).reshape((total_round, NCOL))
    bM_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bM_col_u').values()).reshape((total_round, NCOL))
    bM_col_x = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bM_col_x').values()).reshape((total_round, NCOL))
    bM_col_y = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bM_col_y').values()).reshape((total_round, NCOL))

    fA_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fA_col_u').values()).reshape((total_round, NCOL))
    fA_col_x = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fA_col_x').values()).reshape((total_round, NCOL))
    fA_col_y = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fA_col_y').values()).reshape((total_round, NCOL))
    bA_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bA_col_u').values()).reshape((total_round, NCOL))
    bA_col_x = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bA_col_x').values()).reshape((total_round, NCOL))
    bA_col_y = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bA_col_y').values()).reshape((total_round, NCOL))

    # add constraints for u-x-y encoding
    for r in range(total_round):
        for j in COL:
            m.addConstr(fM_col_x[r,j] == gp.min_(fM_b[r,:,j].tolist()))
            m.addConstr(fM_col_y[r,j] == gp.min_(fM_r[r,:,j].tolist()))
            m.addConstr(fM_col_u[r,j] == gp.max_(fM_w[r,:,j].tolist()))
            m.addConstr(bM_col_x[r,j] == gp.min_(bM_b[r,:,j].tolist()))
            m.addConstr(bM_col_y[r,j] == gp.min_(bM_r[r,:,j].tolist()))
            m.addConstr(bM_col_u[r,j] == gp.max_(bM_w[r,:,j].tolist()))
            
            m.addConstr(fA_col_x[r,j] == gp.min_(fA_b[r,:,j].tolist()))
            m.addConstr(fA_col_y[r,j] == gp.min_(fA_r[r,:,j].tolist()))
            m.addConstr(fA_col_u[r,j] == gp.max_(fA_w[r,:,j].tolist()))
            m.addConstr(bA_col_x[r,j] == gp.min_(bA_b[r,:,j].tolist()))
            m.addConstr(bA_col_y[r,j] == gp.min_(bA_r[r,:,j].tolist()))
            m.addConstr(bA_col_u[r,j] == gp.max_(bA_w[r,:,j].tolist()))
    
    # define auxiliary vars tracking cost of df at MC operations, cost_fwd is solely for MC, cost_bwd is for XOR_MC
    mc_cost_fwd = np.asarray(m.addVars(total_round, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='MC_Cost_fwd').values()).reshape((total_round, NCOL))
    mc_cost_bwd = np.asarray(m.addVars(total_round, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='MC_Cost_bwd').values()).reshape((total_round, NCOL))
    
    # define auxiliary vars tracking cost of df at Add Key operations in foward direction
    xor_cost_fwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='XOR_Cost_fwd').values()).reshape((total_round+1, NROW, NCOL))
    xor_cost_bwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='XOR_Cost_bwd').values()).reshape((total_round+1, NROW, NCOL))

    # define auxiliary vars trackin cost of df in the key expansion process, unpossible combinations are set to zeros
    key_cost_fwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((Nr, NROW, Nk))
    key_cost_bwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((Nr, NROW, Nk))
    
    # Define final states for meet in the middle
    Meet_fwd_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_b').values()).reshape((NROW, NCOL))
    Meet_fwd_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_r').values()).reshape((NROW, NCOL))
    Meet_fwd_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_g').values()).reshape((NROW, NCOL)) 
    Meet_fwd_w = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_w').values()).reshape((NROW, NCOL)) 
    Meet_bwd_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_b').values()).reshape((NROW, NCOL))
    Meet_bwd_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_r').values()).reshape((NROW, NCOL))
    Meet_bwd_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_g').values()).reshape((NROW, NCOL)) 
    Meet_bwd_w = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_w').values()).reshape((NROW, NCOL)) 
    # add constriants for grey indicators
    for i in ROW:
        for j in COL:
            m.addConstr(Meet_fwd_g[i,j] == gp.and_(Meet_fwd_b[i,j], Meet_fwd_r[i,j]))
            m.addConstr(Meet_fwd_w[i,j] + Meet_fwd_b[i,j] + Meet_fwd_r[i,j] - Meet_fwd_g[i,j] == 1)
            m.addConstr(Meet_bwd_g[i,j] == gp.and_(Meet_bwd_b[i,j], Meet_bwd_r[i,j]))
            m.addConstr(Meet_bwd_w[i,j] + Meet_bwd_b[i,j] + Meet_bwd_r[i,j] - Meet_bwd_g[i,j] == 1)
    
    # define auxiliary vars for computations on degree of matching
    meet_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype=GRB.INTEGER, name='Meet_signed').values())
    meet = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Meet').values())

#### Main Procedure ####
    # add constriants according to the key expansion algorithm
    key_expansion(m, key_size, total_round, key_start_round, K_ini_b, K_ini_r, fKeyS_b, fKeyS_r, bKeyS_b, bKeyS_r, CONST0, key_cost_fwd, key_cost_bwd)

    # initialize the enc states, avoid unknown to maximize performance
    for i in ROW:
        for j in COL:
            m.addConstr(E_ini_r[i, j] == 1) #test
            m.addConstr(E_ini_b[i, j] == 0) #test
            m.addConstr(S_b[start_round, i, j] + S_r[start_round, i, j] >= 1)
            m.addConstr(E_ini_b[i, j] == S_b[start_round, i, j])
            m.addConstr(E_ini_r[i, j] == S_r[start_round, i, j])

    # add constriants according to the encryption algorithm
    for r in range(total_round):
        nr = r + 1   # alias for next round
        
        # special case: meet at last round
        if r == match_round and match_round == total_round - 1:
            print('mat lastr', r)
            for i in ROW:
                for j in COL:
                    # Enter SupP at last round MC state
                    gen_ENT_SupP_rule(m, M_b[r,i,j], M_r[r,i,j], fM_b[r,i,j], fM_r[r,i,j], bM_b[r,i,j], bM_r[r,i,j])
                    # add last round key: AK[lr](storing AT) = id(MC[lr]) XOR KEY[lr]
                    gen_XOR_rule(m, fM_b[r,i,j], fM_r[r,i,j], fK_b[r,i,j], fK_r[r,i,j], fA_b[r,i,j], fA_r[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bM_b[r,i,j], bM_r[r,i,j], bK_b[r,i,j], bK_r[r,i,j], bA_b[r,i,j], bA_r[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                    # add whitening key: SB[0] = AK[lr](storing AT) XOR KEY[tr](storing KEY[-1])
                    gen_XOR_rule(m, fA_b[r,i,j], fA_r[r,i,j], fK_b[nr,i,j], fK_r[nr,i,j], fS_b[0,i,j], fS_r[0,i,j], xor_cost_fwd[nr,i,j], CONST0)
                    gen_XOR_rule(m, bA_b[r,i,j], bA_r[r,i,j], bK_b[nr,i,j], bK_r[nr,i,j], bS_b[0,i,j], bS_r[0,i,j], CONST0, xor_cost_bwd[nr,i,j])
                    # Exit SupP at Meet_fwd
                    gen_EXT_SupP_rule(m, fS_b[0,i,j], fS_r[0,i,j], bS_b[0,i,j], bS_r[0,i,j], Meet_fwd_b[i,j], Meet_fwd_r[i,j]) 
            # calculate degree of match
            tempMeet = np.asarray(m.addVars(NROW, NCOL, vtype= GRB.BINARY, name='tempMeet').values()).reshape((NROW, NCOL))
            for j in COL:
                for i in ROW:
                    m.addConstr(tempMeet[i,j] == gp.or_(Meet_fwd_w[i,j], S_w[0,i,j]))
                m.addConstr(meet[j] == NROW - gp.quicksum(tempMeet[:,j]))
                m.addConstr(meet_signed[j] == 0)
            continue
        
        # General structure
        # match round
        if r == match_round:
            print('mat', r)  
            for i in ROW:
                for j in COL:
                    # Enter SupP at next round SB state
                    gen_ENT_SupP_rule(m, S_b[nr,i,j], S_r[nr,i,j], fS_b[nr,i,j], fS_r[nr,i,j], bS_b[nr,i,j], bS_r[nr,i,j])
                    # (reverse) AddKey with SupP  
                    gen_XOR_rule(m, fS_b[nr,i,j], fS_r[nr,i,j], fK_b[r,i,j], fK_r[r,i,j], fA_b[r,i,j], fA_r[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bS_b[nr,i,j], bS_r[nr,i,j], bK_b[r,i,j], bK_r[r,i,j], bA_b[r,i,j], bA_r[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                    # Exit SupP to Meet_bwd
                    gen_EXT_SupP_rule(m, fA_b[r,i,j], fA_r[r,i,j], bA_b[r,i,j], bA_r[r,i,j], Meet_bwd_b[i,j], Meet_bwd_r[i,j])
            # meet-in-the-middle for M(Meet_fwd) and Meet_bwd
            for j in COL:
                gen_match_rule(m, M_b[r,:,j], M_r[r,:,j], M_g[r,:,j], Meet_bwd_b[:,j], Meet_bwd_r[:,j], Meet_bwd_g[:,j], meet_signed[j], meet[j])
            continue
        
        # last round
        if r == total_round - 1:
            #continue
            print('lastr', r)
            # MC of last round is skipped, hence no cost in df
            for j in COL:
                m.addConstr(mc_cost_fwd[r, j] == 0)
                m.addConstr(mc_cost_bwd[r, j] == 0)
            if r in fwd:    # enter last round in fwd direction
                for i in ROW:
                    for j in COL:
                        # Enter SupP at last round MC state
                        gen_ENT_SupP_rule(m, M_b[r,i,j], M_r[r,i,j], fM_b[r,i,j], fM_r[r,i,j], bM_b[r,i,j], bM_r[r,i,j])
                        # add last round key: AK[lr](storing AT) = id(MC[lr]) XOR KEY[lr]
                        gen_XOR_rule(m, fM_b[r,i,j], fM_r[r,i,j], fK_b[r,i,j], fK_r[r,i,j], fA_b[r,i,j], fA_r[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                        gen_XOR_rule(m, bM_b[r,i,j], bM_r[r,i,j], bK_b[r,i,j], bK_r[r,i,j], bA_b[r,i,j], bA_r[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                        # add whitening key: SB[0] = AK[lr](storing AT) XOR KEY[tr](storing KEY[-1])
                        gen_XOR_rule(m, fA_b[r,i,j], fA_r[r,i,j], fK_b[nr,i,j], fK_r[nr,i,j], fS_b[0,i,j], fS_r[0,i,j], xor_cost_fwd[nr,i,j], CONST0)
                        gen_XOR_rule(m, bA_b[r,i,j], bA_r[r,i,j], bK_b[nr,i,j], bK_r[nr,i,j], bS_b[0,i,j], bS_r[0,i,j], CONST0, xor_cost_bwd[nr,i,j])
                        # Exit SupP at round 0 SB state
                        gen_EXT_SupP_rule(m, fS_b[0,i,j], fS_r[0,i,j], bS_b[0,i,j], bS_r[0,i,j], S_b[0,i,j], S_r[0,i,j])  
            elif r in bwd:  # enter last round in bwd direction
                for i in ROW:
                    for j in COL:
                        # Enter SupP at round 0 SB state
                        gen_ENT_SupP_rule(m, S_b[0,i,j], S_r[0,i,j], fS_b[0,i,j], fS_r[0,i,j], bS_b[0,i,j], bS_r[0,i,j])
                        # add whitening key: AK[tr-1](storing AT) =  SB[0] XOR KEY[tr](storing KEY[-1])
                        gen_XOR_rule(m, fS_b[0,i,j], fS_r[0,i,j], fK_b[nr,i,j], fK_r[nr,i,j], fA_b[r,i,j], fA_r[r,i,j], xor_cost_fwd[nr,i,j], CONST0)
                        gen_XOR_rule(m, bS_b[0,i,j], bS_r[0,i,j], bK_b[nr,i,j], bK_r[nr,i,j], bA_b[r,i,j], bA_r[r,i,j], CONST0, xor_cost_bwd[nr,i,j])
                        # add last round key: MC[lr] == id(MC[lr]) = AK[lr](storing AT) XOR KEY[lr]
                        gen_XOR_rule(m, fA_b[r,i,j], fA_r[r,i,j], fK_b[r,i,j], fK_r[r,i,j], fM_b[r,i,j], fM_r[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                        gen_XOR_rule(m, bA_b[r,i,j], bA_r[r,i,j], bK_b[r,i,j], bK_r[r,i,j], bM_b[r,i,j], bM_r[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                        # Ext SupP feed the outcome to current MC state
                        gen_EXT_SupP_rule(m, fM_b[r,i,j], fM_r[r,i,j], bM_b[r,i,j], bM_r[r,i,j], M_b[r,i,j], M_r[r,i,j])
            continue 
        
        # forward direction
        if r in fwd:
            # Enter SupP at current round MC state
            print('fwd', r)
            for i in ROW:
                for j in COL:
                    gen_ENT_SupP_rule(m, M_b[r,i,j], M_r[r,i,j], fM_b[r,i,j], fM_r[r,i,j], bM_b[r,i,j], bM_r[r,i,j])
            # MixCol with SupP
            for j in COL:
                #continue
                gen_MC_rule(m, fM_b[r,:,j], fM_r[r,:,j], fM_col_u[r,j], fM_col_x[r,j], fM_col_y[r,j], fA_b[r,:,j], fA_r[r,:,j], mc_cost_fwd[r,j], CONST0)
                gen_MC_rule(m, bM_b[r,:,j], bM_r[r,:,j], bM_col_u[r,j], bM_col_x[r,j], bM_col_y[r,j], bA_b[r,:,j], bA_r[r,:,j], CONST0, mc_cost_bwd[r,j])
            # AddKey with SupP    
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fA_b[r,i,j], fA_r[r,i,j], fK_b[r,i,j], fK_r[r,i,j], fS_b[nr,i,j], fS_r[nr,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bA_b[r,i,j], bA_r[r,i,j], bK_b[r,i,j], bK_r[r,i,j], bS_b[nr,i,j], bS_r[nr,i,j], CONST0, xor_cost_bwd[r,i,j])
            # Ext SupP feed the outcome to next SB state
            for i in ROW:
                for j in COL:
                    gen_EXT_SupP_rule(m, fS_b[nr,i,j], fS_r[nr,i,j], bS_b[nr,i,j], bS_r[nr,i,j], S_b[nr,i,j], S_r[nr,i,j])
            continue
        
        # backward direction
        elif r in bwd:
            print('bwd', r)
            # Enter SupP at next round SB state
            for i in ROW:
                for j in COL:
                    gen_ENT_SupP_rule(m, S_b[nr,i,j], S_r[nr,i,j], fS_b[nr,i,j], fS_r[nr,i,j], bS_b[nr,i,j], bS_r[nr,i,j])
            # (reverse) AddKey with SupP    
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fS_b[nr,i,j], fS_r[nr,i,j], fK_b[r,i,j], fK_r[r,i,j], fA_b[r,i,j], fA_r[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bS_b[nr,i,j], bS_r[nr,i,j], bK_b[r,i,j], bK_r[r,i,j], bA_b[r,i,j], bA_r[r,i,j], CONST0, xor_cost_bwd[r,i,j])
            # (reverse) MixCol with SupP
            for j in COL:
                #continue
                gen_MC_rule(m, fA_b[r,:,j], fA_r[r,:,j], fA_col_u[r,j], fA_col_x[r,j], fA_col_y[r,j], fM_b[r,:,j], fM_r[r,:,j], mc_cost_fwd[r,j], CONST0)
                gen_MC_rule(m, bA_b[r,:,j], bA_r[r,:,j], bA_col_u[r,j], bA_col_x[r,j], bA_col_y[r,j], bM_b[r,:,j], bM_r[r,:,j], CONST0, mc_cost_bwd[r,j])
            # Ext SupP feed the outcome to current MC state
            for i in ROW:
                for j in COL:
                    gen_EXT_SupP_rule(m, fM_b[r,i,j], fM_r[r,i,j], bM_b[r,i,j], bM_r[r,i,j], M_b[r,i,j], M_r[r,i,j])
            continue
        else:
            raise Exception("Irregular Behavior at encryption")
    
    # set objective function
    set_obj(m, E_ini_b, E_ini_r, E_ini_g, K_ini_b, K_ini_r, K_ini_g, mc_cost_fwd, mc_cost_bwd, xor_cost_fwd, xor_cost_bwd, key_cost_fwd, key_cost_bwd, meet)
    
    m.setParam(GRB.Param.PoolSearchMode, 2)
    m.setParam(GRB.Param.PoolSolutions,  1)
    m.setParam(GRB.Param.BestObjStop, 1.999999999)
    m.setParam(GRB.Param.Cutoff, 1)
    #m.setParam(GRB.Param.PoolObjBound,  2)
    m.setParam(GRB.Param.Threads, 8)
    m.optimize()
    
    if not os.path.exists(path= dir):
        os.makedirs(dir)
    
    m.write(dir + m.modelName + '.lp')
    
    if writeSol(m, path=dir):
        solution = displaySol(m, path=dir)
        return (total_round, start_round, match_round, key_start_round), 1, str(solution)
    else:
        return (total_round, start_round, match_round, key_start_round), 0, 'Infeasible'

#solve(192, 9, 3, 8, 3, dir='./RK_SupP/runs/')
solve(192, 9, 4, 0, 0, dir='./RK_SupP/runs/')
#solve(128, 8, 4, 1, 5, dir='./RK_SupP/trails/')
#displaySol(m, path=dir)