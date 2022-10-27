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

def ent_SupP(m: gp.Model, X_b: gp.Var, X_r: gp.Var, fX_b: gp.Var, fX_r: gp.Var, bX_b: gp.Var, bX_r: gp.Var):
    # seperate MC states into superposition: MC[b,r] -> MC_fwd[b,r] + MC_bwd[b,r]
    # truth table: (1,0)->(1,0)+(1,1); 
    #              (0,1)->(1,1)+(0,1); 
    #              (1,1)->(1,1)+(1,1); 
    #              (0,0)->(0,0)+(0,0);

    m.addConstr(fX_b == gp.max_(X_b, X_r))
    m.addConstr(fX_r == X_r)
    m.addConstr(bX_b == X_b)
    m.addConstr(bX_r == gp.max_(X_b, X_r))

def ext_SupP(m: gp.Model, fX_b: gp.Var, fX_r: gp.Var, bX_b: gp.Var, bX_r: gp.Var, X_b: gp.Var, X_r: gp.Var):
    # truth table: (1,0) + (1,1) or (1,0) -> (1,0)
    #              (0,1) + (1,1) or (0,1) -> (0,1)
    #              (1,1) + (1,1) -> (1,1)
    #              otherwise -> (0,0)
    m.addConstr(X_b == gp.min_(fX_b, bX_b))
    m.addConstr(X_r == gp.min_(fX_r, bX_r))

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
def set_obj(m: gp.Model, Nr, Nb, Nk,
    ini_enc_b: np.ndarray, ini_enc_r: np.ndarray, ini_enc_g: np.ndarray, ini_key_b: np.ndarray, ini_key_r: np.ndarray, ini_key_g: np.ndarray, 
    cost_fwd: np.ndarray, cost_bwd: np.ndarray, xor_cost_fwd: np.ndarray, xor_cost_bwd: np.ndarray, 
    key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray, fKS_eq_g, bKS_eq_g,
    fM_Gx: np.ndarray, fM_x: np.ndarray, bM_Gy: np.ndarray, bM_y: np.ndarray, fA_Gx: np.ndarray, fA_x: np.ndarray, bA_Gy: np.ndarray, bA_y: np.ndarray, 
    M_Gxy: np.ndarray, A_Gxy: np.ndarray, meet: np.ndarray):
    
    true_key_cost_fwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='true_key_cost_fwd').values()).reshape((Nr, NROW, Nk))
    true_key_cost_bwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='true_key_cost_bwd').values()).reshape((Nr, NROW, Nk))
    for r in range(Nr):
        for i in range(Nb):
            for j in range(Nk):
                m.addConstr(true_key_cost_fwd[r,i,j] == key_cost_fwd[r,i,j] - fKS_eq_g[r,i,j])
                m.addConstr(true_key_cost_bwd[r,i,j] == key_cost_bwd[r,i,j] - bKS_eq_g[r,i,j])

    df_b = m.addVar(lb=1, vtype=GRB.INTEGER, name="DF_b")
    df_r = m.addVar(lb=1, vtype=GRB.INTEGER, name="DF_r")
    dm = m.addVar(lb=1, vtype=GRB.INTEGER, name="Match")
    obj = m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

    GnD_b = m.addVar(lb=0, vtype=GRB.INTEGER, name="GND_b")
    GnD_r = m.addVar(lb=0, vtype=GRB.INTEGER, name="GND_r")
    GnD_br = m.addVar(lb=0, vtype=GRB.INTEGER, name="GND_br")

    m.addConstr(GnD_b == gp.quicksum(fM_Gx.flatten()) - gp.quicksum(fM_x.flatten()) + gp.quicksum(fA_Gx.flatten()) - gp.quicksum(fA_x.flatten()))
    m.addConstr(GnD_b == gp.quicksum(bM_Gy.flatten()) - gp.quicksum(bM_y.flatten()) + gp.quicksum(bA_Gy.flatten()) - gp.quicksum(bA_y.flatten())) 
    m.addConstr(GnD_br == gp.quicksum(M_Gxy.flatten()) + gp.quicksum(A_Gxy.flatten()))

    m.addConstr(df_b == gp.quicksum(ini_enc_b.flatten()) - gp.quicksum(ini_enc_g.flatten()) + gp.quicksum(ini_key_b.flatten()) - gp.quicksum(ini_key_g.flatten()) - gp.quicksum(cost_fwd.flatten()) - gp.quicksum(xor_cost_fwd.flatten()) - gp.quicksum(true_key_cost_fwd.flatten()) )
    m.addConstr(df_r == gp.quicksum(ini_enc_r.flatten()) - gp.quicksum(ini_enc_g.flatten()) + gp.quicksum(ini_key_r.flatten()) - gp.quicksum(ini_key_g.flatten()) - gp.quicksum(cost_bwd.flatten()) - gp.quicksum(xor_cost_bwd.flatten()) - gp.quicksum(true_key_cost_bwd.flatten()) )
    m.addConstr(dm == gp.quicksum(meet.flatten()))
   
    m.addConstr(obj <= df_b - GnD_r)
    m.addConstr(obj <= df_r - GnD_b)
    m.addConstr(obj <= dm - GnD_b - GnD_r - GnD_br)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

# key expansion function
def key_expansion(m:gp.Model, key_size:int, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, fKeyS_x: np.ndarray, fKeyS_y: np.ndarray, fKeyS_g, fKeyS_c, fKeyS_eq_g, bKeyS_x: np.ndarray, bKeyS_y: np.ndarray, bKeyS_g, bKeyS_c, bKeyS_eq_g, CONST_0: gp.Var, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
    # set key parameters
    Nk = key_size // 32
    Nb = 4
    Nr = math.ceil((total_r + 1)*Nb / Nk)

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
            for y in range(2**level-1, 2**(level+1)-1):
                if len(indices[y]) == 2:
                    [yr,yi] = indices[y]
                    ynode = KeySub[yr,yi]
                else: 
                    [yr,yi,yj] = indices[y]
                    ynode = KeyS[yr,yi,yj]
                if xnode.sameAs(ynode):
                    count += 1
            if count % 2 == 1:
                parents += [xnode]
        
        # return iterable, redundancy removed parent node list
        return list(set(parents))

    Ksub_x = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='Ksub_x').values()).reshape((Nr, NROW))
    Ksub_y = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='Ksub_y').values()).reshape((Nr, NROW))
    fKsub_x = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKsub_x').values()).reshape((Nr, NROW))
    fKsub_y = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKsub_y').values()).reshape((Nr, NROW))
    fKsub_g = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKsub_g').values()).reshape((Nr, NROW))
    fKsub_c = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='fKsub_c').values()).reshape((Nr, NROW))
    bKsub_x = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='bKsub_x').values()).reshape((Nr, NROW))
    bKsub_y = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='bKsub_y').values()).reshape((Nr, NROW))
    bKsub_g = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='bKsub_g').values()).reshape((Nr, NROW))
    bKsub_c = np.asarray(m.addVars(Nr, NROW, vtype= GRB.BINARY, name='bKsub_c').values()).reshape((Nr, NROW))
    for r in range(Nr):
        for i in ROW:
            m.addConstr(fKsub_g[r,i] == gp.min_(fKsub_x[r,i], fKsub_y[r,i]))
            m.addConstr(fKsub_c[r,i] == fKsub_x[r,i] - fKsub_y[r,i])
            m.addConstr(bKsub_g[r,i] == gp.min_(bKsub_x[r,i], bKsub_y[r,i]))
            m.addConstr(bKsub_c[r,i] == bKsub_y[r,i] - bKsub_x[r,i])
    
    m.update()
    for r in range(Nr):
        # initial state
        if r == start_r: 
            for j in range(Nk):
                for i in ROW:
                    m.addConstr(fKeyS_x[r, i, j] == gp.max_(K_ini_b[i, j], K_ini_r[i, j]))
                    m.addConstr(fKeyS_y[r, i, j] == K_ini_r[i, j])
                    m.addConstr(bKeyS_x[r, i, j] == K_ini_b[i, j])
                    m.addConstr(bKeyS_y[r, i, j] == gp.max_(K_ini_b[i, j], K_ini_r[i, j]))
                    m.addConstr(key_cost_bwd[r,i,j] == 0)
                    m.addConstr(key_cost_fwd[r,i,j] == 0)
            continue
        # fwd direction
        elif r > start_r:
            for j in range(Nk):            
                if j == 0:
                    # RotWord
                    pr, pj = r-1, Nk-1
                    fTemp_b, fTemp_r = np.roll(fKeyS_x[pr,:,pj], -1), np.roll(fKeyS_y[pr,:,pj], -1)
                    bTemp_b, bTemp_r = np.roll(bKeyS_x[pr,:,pj], -1), np.roll(bKeyS_y[pr,:,pj], -1)
                    # SubWord
                    for i in ROW:
                        ext_SupP(m, fTemp_b[i], fTemp_r[i], bTemp_b[i], bTemp_r[i], Ksub_x[pr,i], Ksub_y[pr,i])
                        ent_SupP(m, Ksub_x[pr,i], Ksub_y[pr,i], fKsub_x[pr,i], fKsub_y[pr,i], bKsub_x[pr,i], bKsub_y[pr,i])
                    fTemp_b, fTemp_r, fTemp_c = fKsub_x[pr], fKsub_y[pr], fKsub_c[pr] 
                    bTemp_b, bTemp_r, bTemp_c = bKsub_x[pr], bKsub_y[pr], bKsub_c[pr]
                else:               
                    pr, pj = r, j-1
                    fTemp_b, fTemp_r, fTemp_c = fKeyS_x[pr,:,pj], fKeyS_y[pr,:,pj], fKeyS_c[pr,:,pj]
                    bTemp_b, bTemp_r, bTemp_c = bKeyS_x[pr,:,pj], bKeyS_y[pr,:,pj], bKeyS_c[pr,:,pj] 
                qr, qj = r-1, j      # compute round and column params for w[i-Nk]
                for i in ROW:
                    gen_XOR_rule(m, fKeyS_x[qr,i,qj], fKeyS_y[qr,i,qj], fTemp_b[i], fTemp_r[i], fKeyS_x[r,i,j], fKeyS_y[r,i,j], key_cost_fwd[r,i,j], CONST_0)
                    gen_XOR_rule(m, bKeyS_x[qr,i,qj], bKeyS_y[qr,i,qj], bTemp_b[i], bTemp_r[i], bKeyS_x[r,i,j], bKeyS_y[r,i,j], CONST_0, key_cost_bwd[r,i,j])
                    # determine if the cell is a grey cell equivalent with two blue inputs (fwd, red if bwd) (can be reduced to the xor of some grey cells)
                    m.addConstr(fKeyS_eq_g[r,i,j] == gp.min_(find_parents(fKeyS_g,fKsub_g,r,i,j) + [fTemp_c[i], fKeyS_c[qr,i,qj]]))
                    m.addConstr(bKeyS_eq_g[r,i,j] == gp.min_(find_parents(bKeyS_g,bKsub_g,r,i,j) + [bTemp_c[i], bKeyS_c[qr,i,qj]]))
                    # if the cell is grey equivalent (i.e. eq_g==1), we force the key cost as 1 to output a grey cell as wrote in the XOR rule
                    m.addConstr(key_cost_fwd[r,i,j] >= fKeyS_eq_g[r,i,j])
                    m.addConstr(key_cost_bwd[r,i,j] >= bKeyS_eq_g[r,i,j])
                    m.addConstr(fKeyS_g[r,i,j] >= fKeyS_eq_g[r,i,j])  
                    m.addConstr(bKeyS_g[r,i,j] >= bKeyS_eq_g[r,i,j])
        # bwd direction
        elif r < start_r:  
            for j in range(Nk):
                if j == 0:
                    # RotWord
                    pr, pj = r, Nk-1
                    fTemp_b, fTemp_r = np.roll(fKeyS_x[pr,:,pj], -1), np.roll(fKeyS_y[pr,:,pj], -1)
                    bTemp_b, bTemp_r = np.roll(bKeyS_x[pr,:,pj], -1), np.roll(bKeyS_y[pr,:,pj], -1)
                    # SubWord
                    for i in ROW:
                        ext_SupP(m, fTemp_b[i], fTemp_r[i], bTemp_b[i], bTemp_r[i], Ksub_x[pr,i], Ksub_y[pr,i])
                        ent_SupP(m, Ksub_x[pr,i], Ksub_y[pr,i], fKsub_x[pr,i], fKsub_y[pr,i], bKsub_x[pr,i], bKsub_y[pr,i])
                    fTemp_b, fTemp_r, fTemp_c = fKsub_x[pr], fKsub_y[pr], fKsub_c[pr] 
                    bTemp_b, bTemp_r, bTemp_c = bKsub_x[pr], bKsub_y[pr], bKsub_c[pr]
                else:               
                    pr, pj = r+1, j-1
                    fTemp_b, fTemp_r, fTemp_c = fKeyS_x[pr,:,pj], fKeyS_y[pr,:,pj], fKeyS_c[pr,:,pj]
                    bTemp_b, bTemp_r, bTemp_c = bKeyS_x[pr,:,pj], bKeyS_y[pr,:,pj], bKeyS_c[pr,:,pj]  
                qr, qj = r+1, j      # compute round and column params for w[i-Nk]
                for i in ROW:
                    gen_XOR_rule(m, fKeyS_x[qr,i,qj], fKeyS_y[qr,i,qj], fTemp_b[i], fTemp_r[i], fKeyS_x[r,i,j], fKeyS_y[r,i,j], key_cost_fwd[r,i,j], CONST_0)
                    gen_XOR_rule(m, bKeyS_x[qr,i,qj], bKeyS_y[qr,i,qj], bTemp_b[i], bTemp_r[i], bKeyS_x[r,i,j], bKeyS_y[r,i,j], CONST_0, key_cost_bwd[r,i,j])
                    # determine if the cell is a grey cell equivalent (can be reduced to the xor of some grey cells)
                    m.addConstr(fKeyS_eq_g[r,i,j] == gp.min_(find_parents(fKeyS_g,fKsub_g,r,i,j) + [fTemp_c[i], fKeyS_c[qr,i,qj]]))
                    m.addConstr(bKeyS_eq_g[r,i,j] == gp.min_(find_parents(bKeyS_g,bKsub_g,r,i,j) + [bTemp_c[i], bKeyS_c[qr,i,qj]]))
                    # if the cell is grey equivalent (i.e. eq_g==1), we force the key cost as 1 to output a grey cell as wrote in the XOR rule
                    m.addConstr(key_cost_fwd[r,i,j] >= fKeyS_eq_g[r,i,j])
                    m.addConstr(key_cost_bwd[r,i,j] >= bKeyS_eq_g[r,i,j])
                    m.addConstr(fKeyS_g[r,i,j] >= fKeyS_eq_g[r,i,j])  
                    m.addConstr(bKeyS_g[r,i,j] >= bKeyS_eq_g[r,i,j])
        
        else:
            raise Exception("Irregular Behavior at key schedule")
        m.update()

# simple txt display of the solution
def displaySol(key_size:int, total_round:int, enc_start_round:int, match_round:int, key_start_round:int, model_name:str, sol_i:int, dir):
    def color(b,r):
        if b==1 and r==0:
            return 'b'
        if b==0 and r==1:
            return 'r'
        if b==1 and r==1:
            return 'g'
        if b==0 and r==0:
            return 'w'

    solFile = open(dir + model_name + '_sol_' + str(sol_i) + '.sol', 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
 
    if enc_start_round < match_round:
        fwd = list(range(enc_start_round, match_round))
        bwd = list(range(match_round + 1, total_round)) + list(range(0, enc_start_round))
    else:
        bwd = list(range(match_round + 1, enc_start_round))
        fwd = list(range(enc_start_round, total_round)) + list(range(0, match_round))

    Nb = NCOL
    Nk = key_size // NBYTE
    Nr = math.ceil((total_round + 1)*Nb / Nk)
        
    SB_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    SB_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fSB_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fSB_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bSB_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bSB_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    MC_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fMC_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fMC_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bMC_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bMC_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fAK_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fAK_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAK_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAK_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fKEY_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fKEY_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKEY_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKEY_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

    fKSch_x= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    fKSch_y= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKSch_x= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKSch_y= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)

    KSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    KSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    fKSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    fKSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    bKSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    bKSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    
    Key_cost_fwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    Key_cost_bwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    xor_cost_fwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    xor_cost_bwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    mc_cost_fwd = np.ndarray(shape=(total_round, NCOL), dtype=int)
    mc_cost_bwd = np.ndarray(shape=(total_round, NCOL), dtype=int)

    ini_enc_x = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_y = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_g = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_x = np.ndarray(shape=(NROW, Nk), dtype=int)
    ini_key_y = np.ndarray(shape=(NROW, Nk), dtype=int)
    ini_key_g = np.ndarray(shape=(NROW, Nk), dtype=int)
    
    Meet_fwd_x = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_fwd_y = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_bwd_x = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_bwd_y = np.ndarray(shape=(NROW, NCOL), dtype=int)
    meet =  np.ndarray(shape=(NCOL), dtype=int)
    meet_s =  np.ndarray(shape=(NCOL), dtype=int)

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                SB_x[r,i,j]=Sol["S_x[%d,%d,%d]" %(r,i,j)]
                SB_y[r,i,j]=Sol["S_y[%d,%d,%d]" %(r,i,j)]
                fSB_x[r,i,j]=Sol["fS_x[%d,%d,%d]" %(r,i,j)]
                fSB_y[r,i,j]=Sol["fS_y[%d,%d,%d]" %(r,i,j)]
                bSB_x[r,i,j]=Sol["bS_x[%d,%d,%d]" %(r,i,j)]
                bSB_y[r,i,j]=Sol["bS_y[%d,%d,%d]" %(r,i,j)]

                fAK_x[r,i,j]=Sol["fA_x[%d,%d,%d]" %(r,i,j)]
                fAK_y[r,i,j]=Sol["fA_y[%d,%d,%d]" %(r,i,j)]
                bAK_x[r,i,j]=Sol["bA_x[%d,%d,%d]" %(r,i,j)]
                bAK_y[r,i,j]=Sol["bA_y[%d,%d,%d]" %(r,i,j)]

                fMC_x[r,i,j]=Sol["fM_x[%d,%d,%d]" %(r,i,j)]
                fMC_y[r,i,j]=Sol["fM_y[%d,%d,%d]" %(r,i,j)]
                bMC_x[r,i,j]=Sol["bM_x[%d,%d,%d]" %(r,i,j)]
                bMC_y[r,i,j]=Sol["bM_y[%d,%d,%d]" %(r,i,j)]

    for ri in range(total_round):
        for i in ROW:
            for j in COL:
                MC_x[ri, i, j] = SB_x[ri, i, (j + i)%NCOL]
                MC_y[ri, i, j] = SB_y[ri, i, (j + i)%NCOL]

    
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_round):
        for j in COL:
            for i in ROW:
                fKEY_x[r,i,j] = Sol["fKS_x[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                fKEY_y[r,i,j] = Sol["fKS_y[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bKEY_x[r,i,j] = Sol["bKS_x[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bKEY_y[r,i,j] = Sol["bKS_y[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0
    
    for r in range(Nr):
        for i in ROW:
            for j in range(Nk):
                fKSch_x[r,i,j] = Sol["fKS_x[%d,%d,%d]" %(r,i,j)]
                fKSch_y[r,i,j] = Sol["fKS_y[%d,%d,%d]" %(r,i,j)]
                bKSch_x[r,i,j] = Sol["bKS_x[%d,%d,%d]" %(r,i,j)]
                bKSch_y[r,i,j] = Sol["bKS_y[%d,%d,%d]" %(r,i,j)]

                Key_cost_fwd[r,i,j] = Sol["true_key_cost_fwd[%d,%d,%d]" %(r,i,j)]
                Key_cost_bwd[r,i,j] = Sol["true_key_cost_bwd[%d,%d,%d]" %(r,i,j)]
    
    for r in range(Nr):
        for i in ROW:
            KSub_x[r,i] = Sol["Ksub_x[%d,%d]" %(r,i)]
            KSub_y[r,i] = Sol["Ksub_y[%d,%d]" %(r,i)]
            fKSub_x[r,i] = Sol["fKsub_x[%d,%d]" %(r,i)]
            fKSub_y[r,i] = Sol["fKsub_y[%d,%d]" %(r,i)]
            bKSub_x[r,i] = Sol["bKsub_x[%d,%d]" %(r,i)]
            bKSub_y[r,i] = Sol["bKsub_y[%d,%d]" %(r,i)]
    
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
            ini_enc_x[i,j] = Sol["E_ini_x[%d,%d]" %(i,j)]
            ini_enc_y[i,j] = Sol["E_ini_y[%d,%d]" %(i,j)]
            ini_enc_g[i,j] = Sol["E_ini_g[%d,%d]" %(i,j)]
    
    for i in ROW:
        for j in range(Nk):
            ini_key_x[i,j] = Sol["K_ini_x[%d,%d]" %(i,j)]
            ini_key_y[i,j] = Sol["K_ini_y[%d,%d]" %(i,j)]
            ini_key_g[i,j] = Sol["K_ini_g[%d,%d]" %(i,j)]

    for i in ROW:
        for j in COL:
            Meet_fwd_x[i,j] = Sol["Meet_fwd_x[%d,%d]" %(i,j)]
            Meet_fwd_y[i,j] = Sol["Meet_fwd_y[%d,%d]" %(i,j)]
            Meet_bwd_x[i,j] = Sol["Meet_bwd_x[%d,%d]" %(i,j)]
            Meet_bwd_y[i,j] = Sol["Meet_bwd_y[%d,%d]" %(i,j)]
    
    for j in COL:
        meet[j] = Sol["Meet[%d]" %j]
        meet_s[j] = Sol["Meet_signed[%d]" %j]

    ini_df_enc_b = np.sum(ini_enc_x[:,:]) - np.sum(ini_enc_g[:,:])
    ini_df_enc_r = np.sum(ini_enc_y[:,:]) - np.sum(ini_enc_g[:,:])

    ini_df_key_b = np.sum(ini_key_x[:,:]) - np.sum(ini_key_g[:,:])
    ini_df_key_r = np.sum(ini_key_y[:,:]) - np.sum(ini_key_g[:,:])

    DF_b = Sol["DF_b"]
    DF_r = Sol["DF_r"]
    Match = Sol["Match"]
    GnD_b = Sol["GND_b"]
    GnD_r = Sol["GND_r"]
    GnD_br = Sol["GND_br"]
    Obj = Sol["Obj"]

    f =  open(dir + 'Vis_' + model_name + '_sol_' + str(sol_i) + '.txt', 'w')
    f.write('Model:\n')
    f.write(TAB+ 'Total: ' + str(total_round) +'\n')
    f.write(TAB+ 'Start at: r' + str(enc_start_round) +'\n')
    f.write(TAB+ 'Meet at: r' + str(match_round) +'\n')
    f.write(TAB+ 'KEY start at: r' + str(key_start_round) +'\n')
    f.write('\nInitialization:\n')
    f.write(TAB+'ENC FWD: ' + str(ini_df_enc_b) + '\n' + TAB+ 'ENC BWD: ' + str(ini_df_enc_r) + '\n')
    f.write(TAB+'KEY FWD: ' + str(ini_df_key_b) + '\n' + TAB+ 'KEY BWD: ' + str(ini_df_key_r) + '\n')
    f.write('\nSolution:\n'+TAB+'Obj := min{DF_b=%d - GnD_b=%d, DF_r=%d - GnD_r=%d, Match=%d - GnD_b - GnD_r - GnD_br=%d} = %d' %(DF_b, GnD_b, DF_r, GnD_r, Match, GnD_br, Obj) + '\n')
    f.write('\nVisualization:\n')
    
    for r in range(total_round):
        header = "r%d  " %r 
        
        if r == match_round:
            header+= 'mat -><-'
        elif r in fwd:
            header+= 'fwd --->'
        elif r in bwd:
            header+= 'bwd <---'
        if r == enc_start_round:
            header+=TAB*2 + 'ENC_start'
        
        f.write(header + '\n')
        nr = (r+1)%total_round
        
        line1 = ''
        line2 = ''
        for i in ROW:
            SB, MC, fMC, bMC, fAK, bAK, fKEY, bKEY, fSB, bSB, SBN, KEY = '','','','','','','','','','','','' 
            for j in COL:
                SB+=color(SB_x[r,i,j], SB_y[r,i,j])
                MC+=color(MC_x[r,i,j], MC_y[r,i,j])
                fMC+=color(fMC_x[r,i,j], fMC_y[r,i,j])
                bMC+=color(bMC_x[r,i,j], bMC_y[r,i,j])
                fAK+=color(fAK_x[r,i,j], fAK_y[r,i,j])
                bAK+=color(bAK_x[r,i,j], bAK_y[r,i,j])
                
                fKEY+=color(fKEY_x[r,i,j], fKEY_y[r,i,j])
                bKEY+=color(bKEY_x[r,i,j], bKEY_y[r,i,j])

                fSB+=color(fSB_x[nr,i,j], fSB_y[nr,i,j])
                bSB+=color(bSB_x[nr,i,j], bSB_y[nr,i,j])
                SBN+=color(SB_x[nr,i,j], SB_y[nr,i,j])
            
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
                    MC+=color(MC_x[r,i,j], MC_y[r,i,j])
                    Meet_B+=color(Meet_bwd_x[i,j], Meet_bwd_y[i,j])
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
            fAT +=color(fAK_x[r,i,j], fAK_y[r,i,j])
            bAT +=color(bAK_x[r,i,j], bAK_y[r,i,j])
            fKEY +=color(fKEY_x[r,i,j], fKEY_y[r,i,j])
            bKEY +=color(bKEY_x[r,i,j], bKEY_y[r,i,j])
            fNSB += color(fSB_x[0,i,j], fSB_y[0,i,j])
            bNSB += color(bSB_x[0,i,j], bSB_y[0,i,j])
            NSB += color(SB_x[nr,i,j], SB_y[nr,i,j])
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
                SB +=color(SB_x[0,i,j], SB_y[0,i,j])
                MF +=color(Meet_fwd_x[i,j], Meet_fwd_y[i,j])
            f.write(MF+ TAB*2 + SB + '\n')

    f.write('\n'+'Key Schedule: starts at r'+str(key_start_round)+'\n')
    
    for r in range(Nr):
        f.write('KEY_SCHEDULE_'+str(r)+'\n')
        line1 = ''
        line2 = ''
        for i in ROW:
            for j in range(Nk):
                line1 += color(fKSch_x[r,i,j], fKSch_y[r,i,j])
                line2 += color(bKSch_x[r,i,j], bKSch_y[r,i,j])
                if j == Nk-1 and r!= 0 and r!=Nr-1:
                    line1 += TAB + color(KSub_x[r,(i-1)%4], KSub_y[r,(i-1)%4]) + TAB + color(KSub_x[r,i], KSub_y[r,i]) + TAB + color(fKSub_x[r,i], fKSub_y[r,i])
                    line2 += TAB + ' '                                         + TAB + ' '                             + TAB + color(bKSub_x[r,i], bKSub_y[r,i])
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

# interable solve function with parameters
def solve(key_size:int, total_round:int, enc_start_round:int, match_round:int, key_start_round:int, dir):
    # define optimization model
    m = gp.Model('AES%dRK_%dr_ENC_r%d_Meet_r%d_KEY_r%d' % (key_size, total_round, enc_start_round, match_round, key_start_round))
    
    # Calculate Nk for key schedule
    Nb = NCOL
    Nk = key_size // NBYTE
    Nr = math.ceil((total_round + 1)*Nb / Nk)
    
    # assign forward and backward rounds, excluding match round and last round
    if enc_start_round < match_round:
        fwd = list(range(enc_start_round, match_round))
        bwd = list(range(match_round+1, total_round)) + list(range(0, enc_start_round))
    else:
        bwd = list(range(match_round+1, enc_start_round))
        fwd = list(range(enc_start_round, total_round)) + list(range(0, match_round))

#### Define Variables ####
    # define a constant zero, to force cost of df as 0
    CONST0 = m.addVar(vtype = GRB.BINARY, name='Const0')
    m.addConstr(CONST0 == 0)

    # define vars to track the start state of Encryption states (E) 
    E_ini_x = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='E_ini_x').values()).reshape((NROW, NCOL))
    E_ini_y = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='E_ini_y').values()).reshape((NROW, NCOL))
    E_ini_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='E_ini_g').values()).reshape((NROW, NCOL)) 
    # add constriants for grey indicators
    for i in ROW:
        for j in COL:
            m.addConstr(E_ini_x[i,j] + E_ini_y[i,j] >= 1)
            m.addConstr(E_ini_g[i,j] == gp.min_(E_ini_x[i,j], E_ini_y[i,j]))

    # define vars to track the start state of Key states (K)  
    K_ini_x = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_x').values()).reshape((NROW, Nk))
    K_ini_y = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_y').values()).reshape((NROW, Nk))
    K_ini_g = np.asarray(m.addVars(NROW, Nk, vtype=GRB.BINARY, name='K_ini_g').values()).reshape((NROW, Nk))
    # add constraints for grey indicators
    for i in ROW:
        for j in range(Nk):  
            m.addConstr(K_ini_x[i,j] + K_ini_y[i,j] >= 1)      
            m.addConstr(K_ini_g[i,j] == gp.min_(K_ini_x[i,j], K_ini_y[i,j]))

    # define vars storing the SB state at each round with encoding scheme
    S_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_x').values()).reshape((total_round, NROW, NCOL))
    S_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_y').values()).reshape((total_round, NROW, NCOL))
    S_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_g').values()).reshape((total_round, NROW, NCOL))
    S_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='S_w').values()).reshape((total_round, NROW, NCOL))
    # add constraints for grey and white indicators
    for r in range(total_round):
        for i in ROW:
            for j in COL:
                m.addConstr(S_g[r,i,j] == gp.min_(S_x[r,i,j], S_y[r,i,j]))
                m.addConstr(S_w[r,i,j] + S_x[r,i,j] + S_y[r,i,j] - S_g[r,i,j] == 1)

    # create alias storing the MC state at each round with encoding scheme
    M_x = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    M_y = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    M_g = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    M_w = np.ndarray(shape= (total_round, NROW, NCOL), dtype= gp.Var)
    # match the cells with alias through shift rows
    for r in range(total_round):
        for i in ROW:
            for j in COL:   
                M_x[r,i,j] = S_x[r,i,(j+i)%NCOL]
                M_y[r,i,j] = S_y[r,i,(j+i)%NCOL]
                M_g[r,i,j] = S_g[r,i,(j+i)%NCOL]
                M_w[r,i,j] = S_w[r,i,(j+i)%NCOL]

    # define MC states with superposition, fM for MC in fwd direction, bM for MC in bwd direction
    fM_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_x').values()).reshape((total_round, NROW, NCOL))
    fM_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_y').values()).reshape((total_round, NROW, NCOL))
    fM_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_g').values()).reshape((total_round, NROW, NCOL))
    fM_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_w').values()).reshape((total_round, NROW, NCOL))
    bM_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_x').values()).reshape((total_round, NROW, NCOL))
    bM_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_y').values()).reshape((total_round, NROW, NCOL))
    bM_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_g').values()).reshape((total_round, NROW, NCOL))
    bM_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_w').values()).reshape((total_round, NROW, NCOL))

    # define vars storing the Add key state with superposition at each round with encoding scheme
    fA_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_x').values()).reshape((total_round, NROW, NCOL))
    fA_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_y').values()).reshape((total_round, NROW, NCOL))
    fA_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_g').values()).reshape((total_round, NROW, NCOL))
    fA_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_w').values()).reshape((total_round, NROW, NCOL))
    bA_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_x').values()).reshape((total_round, NROW, NCOL))
    bA_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_y').values()).reshape((total_round, NROW, NCOL))
    bA_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_g').values()).reshape((total_round, NROW, NCOL))
    bA_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_w').values()).reshape((total_round, NROW, NCOL))

    # define vars storing the state after adding the key with superposition at each round with encoding scheme
    fS_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_x').values()).reshape((total_round, NROW, NCOL))
    fS_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_y').values()).reshape((total_round, NROW, NCOL))
    fS_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_g').values()).reshape((total_round, NROW, NCOL))
    fS_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_w').values()).reshape((total_round, NROW, NCOL))
    bS_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_x').values()).reshape((total_round, NROW, NCOL))
    bS_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_y').values()).reshape((total_round, NROW, NCOL))
    bS_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_g').values()).reshape((total_round, NROW, NCOL))
    bS_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_w').values()).reshape((total_round, NROW, NCOL))

    # bluk add grey and white constriants
    for r in range(total_round):
        for i in ROW:
            for j in COL:             
                m.addConstr(fM_g[r,i,j] == gp.min_(fM_x[r,i,j], fM_y[r,i,j]))
                m.addConstr(fM_w[r,i,j] + fM_x[r,i,j] + fM_y[r,i,j] - fM_g[r,i,j] == 1)
                m.addConstr(bM_g[r,i,j] == gp.min_(bM_x[r,i,j], bM_y[r,i,j]))
                m.addConstr(bM_w[r,i,j] + bM_x[r,i,j] + bM_y[r,i,j] - bM_g[r,i,j] == 1)
                
                m.addConstr(fA_g[r,i,j] == gp.min_(fA_x[r,i,j], fA_y[r,i,j]))
                m.addConstr(fA_w[r,i,j] + fA_x[r,i,j] + fA_y[r,i,j] - fA_g[r,i,j] == 1)
                m.addConstr(bA_g[r,i,j] == gp.min_(bA_x[r,i,j], bA_y[r,i,j]))
                m.addConstr(bA_w[r,i,j] + bA_x[r,i,j] + bA_y[r,i,j] - bA_g[r,i,j] == 1)

                m.addConstr(fS_g[r,i,j] == gp.min_(fS_x[r,i,j], fS_y[r,i,j]))
                m.addConstr(fS_w[r,i,j] + fS_x[r,i,j] + fS_y[r,i,j] - fS_g[r,i,j] == 1)
                m.addConstr(bS_g[r,i,j] == gp.min_(bS_x[r,i,j], bS_y[r,i,j]))
                m.addConstr(bS_w[r,i,j] + bS_x[r,i,j] + bS_y[r,i,j] - bS_g[r,i,j] == 1) 

    # define GnD vars with constraints
    fM_Gx = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_Gb').values()).reshape((total_round, NROW, NCOL))
    bM_Gy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_Gr').values()).reshape((total_round, NROW, NCOL))
    M_Gxy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='M_Gbr').values()).reshape((total_round, NROW, NCOL))

    fA_Gx = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fA_Gb').values()).reshape((total_round, NROW, NCOL))
    bA_Gy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bA_Gr').values()).reshape((total_round, NROW, NCOL))
    A_Gxy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='A_Gbr').values()).reshape((total_round, NROW, NCOL))
    
    # GnD rules (WLOG, fwd dir): we have in SupP w=0<=>x=1, w=1<=>x=0
    # when a cell is not white, it cannot be guessed (w=0 <=> x=1 -> Gx=1)
    # when a cell is white, it could be guessed (w=1 <=> x=0 ->Gx=0/1)
    # Hence, we have Gx>=x, whenever Gx>x, the bit is guessed, the cost of df in GnD will be: sum(Gx-x)
    # In GnD mode, Gx will be used in MC instead of x
    # if a cell is guessed in BiDr, then the cell in both dir must be white (fXw=bXw=1), and both are guessed to be non-zero value (Gx=Gy=1)
    for r in range(total_round):
        for i in ROW:
            for j in COL: 
                m.addConstr(fM_Gx[r,i,j] >= fM_x[r,i,j])
                m.addConstr(bM_Gy[r,i,j] >= bM_y[r,i,j])
                m.addConstr(M_Gxy[r,i,j] == gp.min_(fM_Gx[r,i,j], bM_Gy[r,i,j], fM_w[r,i,j], bM_w[r,i,j]))

                m.addConstr(fA_Gx[r,i,j] >= fA_x[r,i,j])
                m.addConstr(bA_Gy[r,i,j] >= bA_y[r,i,j])
                m.addConstr(A_Gxy[r,i,j] == gp.min_(fA_Gx[r,i,j], bA_Gy[r,i,j], fA_w[r,i,j], bA_w[r,i,j]))

    # define vars storing the key state in key schedule (the long key), in total Nr rounds, with shape NROW*Nk
    fKS_x = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_x').values()).reshape((Nr, NROW, Nk))
    fKS_y = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_y').values()).reshape((Nr, NROW, Nk))
    fKS_g = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_g').values()).reshape((Nr, NROW, Nk))
    fKS_c = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_c').values()).reshape((Nr, NROW, Nk))
    fKS_eq_g = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='fKS_eq_g').values()).reshape((Nr, NROW, Nk))
    bKS_x = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKS_x').values()).reshape((Nr, NROW, Nk))
    bKS_y = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKS_y').values()).reshape((Nr, NROW, Nk))
    bKS_g = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKS_g').values()).reshape((Nr, NROW, Nk))
    bKS_c = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKS_c').values()).reshape((Nr, NROW, Nk))
    bKS_eq_g = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='bKS_eq_g').values()).reshape((Nr, NROW, Nk))
    
    # define grey cell encodings
    for r in range(Nr):
        for i in ROW:
            for j in range(Nk): 
                m.addConstr(fKS_g[r,i,j] == gp.min_(fKS_x[r,i,j], fKS_y[r,i,j]))
                m.addConstr(fKS_c[r,i,j] == fKS_x[r,i,j] - fKS_y[r,i,j])
                m.addConstr(bKS_g[r,i,j] == gp.min_(bKS_x[r,i,j], bKS_y[r,i,j]))
                m.addConstr(bKS_c[r,i,j] == bKS_y[r,i,j] - bKS_x[r,i,j])
    
    # create alias storing the round keys with SupP
    fK_x = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    fK_y = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    bK_x = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    bK_y = np.ndarray(shape= (total_round + 1, NROW, NCOL), dtype= gp.Var)
    # match the states in key schedule to round keys alias
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_round):
        for j in COL:
            for i in ROW:
                fK_x[r,i,j] = fKS_x[KeyS_r,i,KeyS_j]
                fK_y[r,i,j] = fKS_y[KeyS_r,i,KeyS_j]
                bK_x[r,i,j] = bKS_x[KeyS_r,i,KeyS_j]
                bK_y[r,i,j] = bKS_y[KeyS_r,i,KeyS_j]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0

    # define vars for columnwise encoding for MixCol input, including MC(fwd) and AK(bwd)
    fM_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fM_col_u').values()).reshape((total_round, NCOL))
    fM_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fM_col_v').values()).reshape((total_round, NCOL))
    fM_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fM_col_w').values()).reshape((total_round, NCOL))
    bM_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bM_col_u').values()).reshape((total_round, NCOL))
    bM_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bM_col_v').values()).reshape((total_round, NCOL))
    bM_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bM_col_w').values()).reshape((total_round, NCOL))

    fA_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fA_col_u').values()).reshape((total_round, NCOL))
    fA_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fA_col_v').values()).reshape((total_round, NCOL))
    fA_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fA_col_w').values()).reshape((total_round, NCOL))
    bA_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bA_col_u').values()).reshape((total_round, NCOL))
    bA_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bA_col_v').values()).reshape((total_round, NCOL))
    bA_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bA_col_w').values()).reshape((total_round, NCOL))

    # add constraints for u-x-y encoding
    for r in range(total_round):
        for j in COL:
            m.addConstr(fM_col_v[r,j] == gp.min_(fM_Gx[r,:,j].tolist()))
            m.addConstr(fM_col_w[r,j] == gp.min_(fM_y[r,:,j].tolist()))
            m.addConstr(fM_col_u[r,j] == gp.max_(fM_w[r,:,j].tolist()))
            m.addConstr(bM_col_v[r,j] == gp.min_(bM_x[r,:,j].tolist()))
            m.addConstr(bM_col_w[r,j] == gp.min_(bM_Gy[r,:,j].tolist()))
            m.addConstr(bM_col_u[r,j] == gp.max_(bM_w[r,:,j].tolist()))
            
            m.addConstr(fA_col_v[r,j] == gp.min_(fA_Gx[r,:,j].tolist()))
            m.addConstr(fA_col_w[r,j] == gp.min_(fA_y[r,:,j].tolist()))
            m.addConstr(fA_col_u[r,j] == gp.max_(fA_w[r,:,j].tolist()))
            m.addConstr(bA_col_v[r,j] == gp.min_(bA_x[r,:,j].tolist()))
            m.addConstr(bA_col_w[r,j] == gp.min_(bA_Gy[r,:,j].tolist()))
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
    Meet_fwd_x = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_x').values()).reshape((NROW, NCOL))
    Meet_fwd_y = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_y').values()).reshape((NROW, NCOL))
    Meet_fwd_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_g').values()).reshape((NROW, NCOL)) 
    Meet_fwd_w = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_fwd_w').values()).reshape((NROW, NCOL)) 
    Meet_bwd_x = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_x').values()).reshape((NROW, NCOL))
    Meet_bwd_y = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_y').values()).reshape((NROW, NCOL))
    Meet_bwd_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_g').values()).reshape((NROW, NCOL)) 
    Meet_bwd_w = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_bwd_w').values()).reshape((NROW, NCOL)) 
    # add constriants for grey indicators
    for i in ROW:
        for j in COL:
            m.addConstr(Meet_fwd_g[i,j] == gp.min_(Meet_fwd_x[i,j], Meet_fwd_y[i,j]))
            m.addConstr(Meet_fwd_w[i,j] + Meet_fwd_x[i,j] + Meet_fwd_y[i,j] - Meet_fwd_g[i,j] == 1)
            m.addConstr(Meet_bwd_g[i,j] == gp.min_(Meet_bwd_x[i,j], Meet_bwd_y[i,j]))
            m.addConstr(Meet_bwd_w[i,j] + Meet_bwd_x[i,j] + Meet_bwd_y[i,j] - Meet_bwd_g[i,j] == 1)
    
    # define auxiliary vars for computations on degree of matching
    meet_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype=GRB.INTEGER, name='Meet_signed').values())
    meet = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Meet').values())

#### Main Procedure ####
    # add constriants according to the key expansion algorithm
    key_expansion(m, key_size, total_round, key_start_round, K_ini_x, K_ini_y, fKS_x, fKS_y, fKS_g, fKS_c, fKS_eq_g, bKS_x, bKS_y, bKS_g, bKS_c, bKS_eq_g, CONST0, key_cost_fwd, key_cost_bwd)

    # proposition: fix start state to be all red (WLOG), in compensate of the efficiency
    for i in ROW:
        for j in COL:
            #continue
            m.addConstr(E_ini_y[i, j] == 1) #test
            m.addConstr(E_ini_x[i, j] == 0) #test

    # test for key schedule
    for i in ROW:
        for j in range(Nk):
            #continue
            if (i==0 and j==2) or (i==2 and j==2):
                m.addConstr(K_ini_y[i, j] == 0) #test
                m.addConstr(K_ini_x[i, j] == 1) #test
                continue
            m.addConstr(K_ini_y[i, j] == 1) #test
            m.addConstr(K_ini_x[i, j] == 0) #test

    # initialize the enc states, avoid unknown to maximize performance
    for i in ROW:
        for j in COL:
            m.addConstr(S_x[enc_start_round, i, j] + S_y[enc_start_round, i, j] >= 1)
            m.addConstr(E_ini_x[i, j] == S_x[enc_start_round, i, j])
            m.addConstr(E_ini_y[i, j] == S_y[enc_start_round, i, j])

    # add constriants according to the encryption algorithm
    for r in range(total_round):
        nr = r + 1   # alias for next round
        
        # special case: meet at last round
        if r == match_round and match_round == total_round - 1:
            print('mat lastr', r)
            for i in ROW:
                for j in COL:
                    # Enter SupP at last round MC state
                    ent_SupP(m, M_x[r,i,j], M_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j])
                    # add last round key: AK[lr](storing AT) = id(MC[lr]) XOR KEY[lr]
                    gen_XOR_rule(m, fM_x[r,i,j], fM_y[r,i,j], fK_x[r,i,j], fK_y[r,i,j], fA_x[r,i,j], fA_y[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bM_x[r,i,j], bM_y[r,i,j], bK_x[r,i,j], bK_y[r,i,j], bA_x[r,i,j], bA_y[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                    # add whitening key: SB[0] = AK[lr](storing AT) XOR KEY[tr](storing KEY[-1])
                    gen_XOR_rule(m, fA_x[r,i,j], fA_y[r,i,j], fK_x[nr,i,j], fK_y[nr,i,j], fS_x[0,i,j], fS_y[0,i,j], xor_cost_fwd[nr,i,j], CONST0)
                    gen_XOR_rule(m, bA_x[r,i,j], bA_y[r,i,j], bK_x[nr,i,j], bK_y[nr,i,j], bS_x[0,i,j], bS_y[0,i,j], CONST0, xor_cost_bwd[nr,i,j])
                    # Exit SupP at Meet_fwd
                    ext_SupP(m, fS_x[0,i,j], fS_y[0,i,j], bS_x[0,i,j], bS_y[0,i,j], Meet_fwd_x[i,j], Meet_fwd_y[i,j]) 
            # calculate degree of match
            tempMeet = np.asarray(m.addVars(NROW, NCOL, vtype= GRB.BINARY, name='tempMeet').values()).reshape((NROW, NCOL))
            for j in COL:
                for i in ROW:
                    m.addConstr(tempMeet[i,j] == gp.max_(Meet_fwd_w[i,j], S_w[0,i,j]))
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
                    ent_SupP(m, S_x[nr,i,j], S_y[nr,i,j], fS_x[nr,i,j], fS_y[nr,i,j], bS_x[nr,i,j], bS_y[nr,i,j])
                    # (reverse) AddKey with SupP  
                    gen_XOR_rule(m, fS_x[nr,i,j], fS_y[nr,i,j], fK_x[r,i,j], fK_y[r,i,j], fA_x[r,i,j], fA_y[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bS_x[nr,i,j], bS_y[nr,i,j], bK_x[r,i,j], bK_y[r,i,j], bA_x[r,i,j], bA_y[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                    # Exit SupP to Meet_bwd
                    ext_SupP(m, fA_x[r,i,j], fA_y[r,i,j], bA_x[r,i,j], bA_y[r,i,j], Meet_bwd_x[i,j], Meet_bwd_y[i,j])
            # meet-in-the-middle for M(Meet_fwd) and Meet_bwd
            for j in COL:
                gen_match_rule(m, M_x[r,:,j], M_y[r,:,j], M_g[r,:,j], Meet_bwd_x[:,j], Meet_bwd_y[:,j], Meet_bwd_g[:,j], meet_signed[j], meet[j])
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
                        ent_SupP(m, M_x[r,i,j], M_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j])
                        # add last round key: AK[lr](storing AT) = id(MC[lr]) XOR KEY[lr]
                        gen_XOR_rule(m, fM_x[r,i,j], fM_y[r,i,j], fK_x[r,i,j], fK_y[r,i,j], fA_x[r,i,j], fA_y[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                        gen_XOR_rule(m, bM_x[r,i,j], bM_y[r,i,j], bK_x[r,i,j], bK_y[r,i,j], bA_x[r,i,j], bA_y[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                        # add whitening key: SB[0] = AK[lr](storing AT) XOR KEY[tr](storing KEY[-1])
                        gen_XOR_rule(m, fA_x[r,i,j], fA_y[r,i,j], fK_x[nr,i,j], fK_y[nr,i,j], fS_x[0,i,j], fS_y[0,i,j], xor_cost_fwd[nr,i,j], CONST0)
                        gen_XOR_rule(m, bA_x[r,i,j], bA_y[r,i,j], bK_x[nr,i,j], bK_y[nr,i,j], bS_x[0,i,j], bS_y[0,i,j], CONST0, xor_cost_bwd[nr,i,j])
                        # Exit SupP at round 0 SB state
                        ext_SupP(m, fS_x[0,i,j], fS_y[0,i,j], bS_x[0,i,j], bS_y[0,i,j], S_x[0,i,j], S_y[0,i,j])  
            elif r in bwd:  # enter last round in bwd direction
                for i in ROW:
                    for j in COL:
                        # Enter SupP at round 0 SB state
                        ent_SupP(m, S_x[0,i,j], S_y[0,i,j], fS_x[0,i,j], fS_y[0,i,j], bS_x[0,i,j], bS_y[0,i,j])
                        # add whitening key: AK[tr-1](storing AT) =  SB[0] XOR KEY[tr](storing KEY[-1])
                        gen_XOR_rule(m, fS_x[0,i,j], fS_y[0,i,j], fK_x[nr,i,j], fK_y[nr,i,j], fA_x[r,i,j], fA_y[r,i,j], xor_cost_fwd[nr,i,j], CONST0)
                        gen_XOR_rule(m, bS_x[0,i,j], bS_y[0,i,j], bK_x[nr,i,j], bK_y[nr,i,j], bA_x[r,i,j], bA_y[r,i,j], CONST0, xor_cost_bwd[nr,i,j])
                        # add last round key: MC[lr] == id(MC[lr]) = AK[lr](storing AT) XOR KEY[lr]
                        gen_XOR_rule(m, fA_x[r,i,j], fA_y[r,i,j], fK_x[r,i,j], fK_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                        gen_XOR_rule(m, bA_x[r,i,j], bA_y[r,i,j], bK_x[r,i,j], bK_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], CONST0, xor_cost_bwd[r,i,j])
                        # Ext SupP feed the outcome to current MC state
                        ext_SupP(m, fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], M_x[r,i,j], M_y[r,i,j])
            continue 
        
        # forward direction
        if r in fwd:
            # Enter SupP at current round MC state
            print('fwd', r)
            for i in ROW:
                for j in COL:
                    ent_SupP(m, M_x[r,i,j], M_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j])
            # MixCol with SupP and GnD
            for j in COL:
                #continue
                gen_MC_rule(m, fM_Gx[r,:,j], fM_y[r,:,j], fM_col_u[r,j], fM_col_v[r,j], fM_col_w[r,j], fA_x[r,:,j], fA_y[r,:,j], mc_cost_fwd[r,j], CONST0)
                gen_MC_rule(m, bM_x[r,:,j], bM_Gy[r,:,j], bM_col_u[r,j], bM_col_v[r,j], bM_col_w[r,j], bA_x[r,:,j], bA_y[r,:,j], CONST0, mc_cost_bwd[r,j])
                for i in ROW:   # fix unused guess bits
                    m.addConstr(fA_x[r,i,j] == fA_Gx[r,i,j])
                    m.addConstr(bA_y[r,i,j] == bA_Gy[r,i,j])
            # AddKey with SupP    
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fA_x[r,i,j], fA_y[r,i,j], fK_x[r,i,j], fK_y[r,i,j], fS_x[nr,i,j], fS_y[nr,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bA_x[r,i,j], bA_y[r,i,j], bK_x[r,i,j], bK_y[r,i,j], bS_x[nr,i,j], bS_y[nr,i,j], CONST0, xor_cost_bwd[r,i,j])
            # Ext SupP and feed the outcome to next SB state
            for i in ROW:
                for j in COL:
                    ext_SupP(m, fS_x[nr,i,j], fS_y[nr,i,j], bS_x[nr,i,j], bS_y[nr,i,j], S_x[nr,i,j], S_y[nr,i,j])
            continue
        
        # backward direction
        elif r in bwd:
            print('bwd', r)
            # Enter SupP at next round SB state
            for i in ROW:
                for j in COL:
                    ent_SupP(m, S_x[nr,i,j], S_y[nr,i,j], fS_x[nr,i,j], fS_y[nr,i,j], bS_x[nr,i,j], bS_y[nr,i,j])
            # (reverse) AddKey with SupP    
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fS_x[nr,i,j], fS_y[nr,i,j], fK_x[r,i,j], fK_y[r,i,j], fA_x[r,i,j], fA_y[r,i,j], xor_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bS_x[nr,i,j], bS_y[nr,i,j], bK_x[r,i,j], bK_y[r,i,j], bA_x[r,i,j], bA_y[r,i,j], CONST0, xor_cost_bwd[r,i,j])
            # (reverse) MixCol with SupP and GnD
            for j in COL:
                #continue
                gen_MC_rule(m, fA_Gx[r,:,j], fA_y[r,:,j], fA_col_u[r,j], fA_col_v[r,j], fA_col_w[r,j], fM_x[r,:,j], fM_y[r,:,j], mc_cost_fwd[r,j], CONST0)
                gen_MC_rule(m, bA_x[r,:,j], bA_Gy[r,:,j], bA_col_u[r,j], bA_col_v[r,j], bA_col_w[r,j], bM_x[r,:,j], bM_y[r,:,j], CONST0, mc_cost_bwd[r,j])
                for i in ROW:   # fix unused guess bits
                    m.addConstr(fM_x[r,i,j] == fM_Gx[r,i,j])
                    m.addConstr(bM_y[r,i,j] == bM_Gy[r,i,j])
            # Ext SupP and feed the outcome to current MC state
            for i in ROW:
                for j in COL:
                    ext_SupP(m, fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], M_x[r,i,j], M_y[r,i,j])
            continue
        else:
            raise Exception("Irregular Behavior at encryption")
    
    # set objective function
    set_obj(m, Nr, Nb, Nk, E_ini_x, E_ini_y, E_ini_g, K_ini_x, K_ini_y, K_ini_g, mc_cost_fwd, mc_cost_bwd, xor_cost_fwd, xor_cost_bwd, key_cost_fwd, key_cost_bwd, fKS_eq_g, bKS_eq_g, fM_Gx, fM_x, bM_Gy, bM_y, fA_Gx, fA_x, bA_Gy, bA_y, M_Gxy, A_Gxy, meet)
    
    m.setParam(GRB.Param.PoolSearchMode, 2)
    m.setParam(GRB.Param.PoolSolutions,  1)
    m.setParam(GRB.Param.BestObjStop, 2.999999999)
   #m.setParam(GRB.Param.Cutoff, 1)
    m.setParam(GRB.Param.Threads, 4)
    m.optimize()
    
    if not os.path.exists(path= dir):
        os.makedirs(dir)
    
    m.write(dir + m.modelName + '.lp')

    if m.SolCount > 0:
        for sol_i in range(m.SolCount):
            m.write(dir + m.modelName + '_sol_' + str(sol_i) + '.sol')
            displaySol(key_size, total_round, enc_start_round, match_round, key_start_round, m.modelName, sol_i, dir)
        return m.SolCount
    else:
        return 0

solve(key_size=192, total_round=9, enc_start_round=3, match_round=8, key_start_round=3, dir='./AES_SupP_GnD_RKCorrection/runs/')