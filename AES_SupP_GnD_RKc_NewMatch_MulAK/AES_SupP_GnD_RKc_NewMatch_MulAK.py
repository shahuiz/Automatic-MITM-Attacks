from io import TextIOWrapper
import gurobipy as gp
from gurobipy import GRB
from string import Template
import numpy as np
import re
import os
import math
import copy
import time
from tex_display import tex_display

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
def gen_match_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_g: np.ndarray, in_info: np.ndarray, out_b: np.ndarray, out_r: np.ndarray, out_g: np.ndarray, out_info: np.ndarray, meet):
    meet_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype=GRB.INTEGER, name='Meet_signed').values())
    for j in COL:
        m.addConstr(meet_signed[j] == gp.quicksum(in_b[:,j]) + gp.quicksum(in_r[:,j]) - gp.quicksum(in_g[:,j]) + gp.quicksum(out_info[:,j]) - NROW)
        m.addConstr(meet[j] == gp.max_(meet_signed[j], 0))
    m.update()

def gen_new_match_rule(m: gp.Model, lhs_x, lhs_y, lhs_info: np.ndarray, rhs_x, rhs_y, rhs_info: np.ndarray, meet):
    ind_4blue = np.asarray(m.addVars(NCOL, vtype = GRB.BINARY, name='four_blue_indicator').values())
    ind_4red = np.asarray(m.addVars(NCOL, vtype = GRB.BINARY, name='four_red_indicator').values())
    ind_4same = np.asarray(m.addVars(NCOL, vtype = GRB.BINARY, name='four_same_color_indicator').values())
    
    match_case_2_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype = GRB.INTEGER, name='match_case_2_signed').values())
    match_case_2 = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype = GRB.INTEGER, name='match_case_2').values())
    
    for j in COL:
        #m.addConstr((ind_4same[j] == 1) >> (gp.quicksum(lhs_x[:,j]) + gp.quicksum(rhs_x[:,j]) >= NROW))
        m.addConstr((ind_4blue[j] == 1) >> (gp.quicksum(lhs_x[:,j]) + gp.quicksum(rhs_x[:,j]) >= NROW))
        m.addConstr((ind_4red[j] == 1) >> (gp.quicksum(lhs_y[:,j]) + gp.quicksum(rhs_y[:,j]) >= NROW))
        m.addConstr(ind_4same[j] == gp.max_(ind_4blue[j], ind_4red[j]))

        m.addConstr(match_case_2_signed[j] == gp.quicksum(lhs_info[:,j]) + gp.quicksum(rhs_info[:,j]) - NROW)
        m.addConstr(match_case_2[j] == ind_4same[j] * match_case_2_signed[j])
        m.addConstr(meet[j] == gp.max_(match_case_2[j], 0))
    m.update()

def gen_combined_match_rule(m:gp.Model, lhs_x, lhs_y, lhs_info, lhs_trace, rhs_x, rhs_y, rhs_info, rhs_trace, meet):
    
    ind_4blue = np.asarray(m.addVars(NCOL, vtype = GRB.BINARY, name='four_blue_indicator').values())
    ind_4red = np.asarray(m.addVars(NCOL, vtype = GRB.BINARY, name='four_red_indicator').values())
    ind_4same = np.asarray(m.addVars(NCOL, vtype = GRB.BINARY, name='four_same_color_indicator').values())

    match_case_1_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype = GRB.INTEGER, name='match_case_1_signed').values())
    match_case_2_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype = GRB.INTEGER, name='match_case_2_signed').values())
    
    match_case_1 = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype = GRB.INTEGER, name='match_case_1').values())
    match_case_2 = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype = GRB.INTEGER, name='match_case_2').values())
    
    for j in COL:
        # basic match rule: lhs have to be pure color, rhs can be linear combination state
        m.addConstr(match_case_1_signed[j] == gp.quicksum(lhs_info[:,j]) + gp.quicksum(rhs_info[:,j]) - NROW)
        m.addConstr(match_case_1[j] == gp.max_(match_case_1_signed[j], 0))
        # additional match rule 1: if has 4 same pure color cells at lhs and rhs of MC, then linear combination could be traced thru S-box
        m.addConstr((ind_4blue[j] == 1) >> (gp.quicksum(lhs_x[:,j]) + gp.quicksum(rhs_x[:,j]) >= NROW))
        m.addConstr((ind_4red[j] == 1) >> (gp.quicksum(lhs_y[:,j]) + gp.quicksum(rhs_y[:,j]) >= NROW))
        m.addConstr(ind_4same[j] == gp.max_(ind_4blue[j], ind_4red[j]))
        # activate the match rule
        m.addConstr(match_case_2_signed[j] == gp.quicksum(lhs_trace[:,j]) + gp.quicksum(rhs_trace[:,j]) - NROW)
        m.addConstr(match_case_2[j] == ind_4same[j] * match_case_2_signed[j])
        # pick the largest df for matching
        m.addConstr(meet[j] == gp.max_(match_case_1[j], match_case_2[j]))
    
    m.update()
    return

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

# interable solve function with parameters
def solve(key_size:int, total_round:int, enc_start_round:int, match_round:int, key_start_round:int, dir):
#### Besic Info ####
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
    # trick: define a constant zero, to force cost of df as 0
    CONST0 = m.addVar(vtype = GRB.BINARY, name='Const0')
    m.addConstr(CONST0 == 0)



    ### State Vars
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

    # define vars storing the MC state at each round with encoding scheme
    M_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='M_x').values()).reshape((total_round, NROW, NCOL))
    M_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='M_y').values()).reshape((total_round, NROW, NCOL))
    M_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='M_g').values()).reshape((total_round, NROW, NCOL))
    M_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='M_w').values()).reshape((total_round, NROW, NCOL))
    # add constraints for grey and white indicators
    for r in range(total_round):
        for i in ROW:
            for j in COL:
                m.addConstr(M_x[r,i,j] == S_x[r,i,(j+i)%NCOL])
                m.addConstr(M_y[r,i,j] == S_y[r,i,(j+i)%NCOL])
                m.addConstr(M_g[r,i,j] == gp.min_(M_x[r,i,j], M_y[r,i,j]))
                m.addConstr(M_w[r,i,j] + M_x[r,i,j] + M_y[r,i,j] - M_g[r,i,j] == 1)

    # define MC states with superposition, fM for MC in fwd direction, bM for MC in bwd direction
    fM_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_x').values()).reshape((total_round, NROW, NCOL))
    fM_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_y').values()).reshape((total_round, NROW, NCOL))
    fM_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_g').values()).reshape((total_round, NROW, NCOL))
    fM_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fM_w').values()).reshape((total_round, NROW, NCOL))
    bM_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_x').values()).reshape((total_round, NROW, NCOL))
    bM_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_y').values()).reshape((total_round, NROW, NCOL))
    bM_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_g').values()).reshape((total_round, NROW, NCOL))
    bM_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bM_w').values()).reshape((total_round, NROW, NCOL))

    # define vars storing the equivalent states with superposition on LHS and RHS of the MixCol state
    # (fM,bM) ---AddKeyLHS---> (fAL,fAR) <---[MixCol]---> (fAR,bAR) <---AddKeyRHS--- (fNSB,bNSB)
    fAL_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAL_x').values()).reshape((total_round, NROW, NCOL))
    fAL_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAL_y').values()).reshape((total_round, NROW, NCOL))
    fAL_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAL_g').values()).reshape((total_round, NROW, NCOL))
    fAL_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAL_w').values()).reshape((total_round, NROW, NCOL))
    bAL_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAL_x').values()).reshape((total_round, NROW, NCOL))
    bAL_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAL_y').values()).reshape((total_round, NROW, NCOL))
    bAL_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAL_g').values()).reshape((total_round, NROW, NCOL))
    bAL_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAL_w').values()).reshape((total_round, NROW, NCOL))  
    
    fAR_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAR_x').values()).reshape((total_round, NROW, NCOL))
    fAR_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAR_y').values()).reshape((total_round, NROW, NCOL))
    fAR_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAR_g').values()).reshape((total_round, NROW, NCOL))
    fAR_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAR_w').values()).reshape((total_round, NROW, NCOL))
    bAR_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAR_x').values()).reshape((total_round, NROW, NCOL))
    bAR_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAR_y').values()).reshape((total_round, NROW, NCOL))
    bAR_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAR_g').values()).reshape((total_round, NROW, NCOL))
    bAR_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAR_w').values()).reshape((total_round, NROW, NCOL))

    # define vars storing the state after adding the key with superposition at each round with encoding scheme
    fS_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_x').values()).reshape((total_round, NROW, NCOL))
    fS_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_y').values()).reshape((total_round, NROW, NCOL))
    fS_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_g').values()).reshape((total_round, NROW, NCOL))
    fS_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fS_w').values()).reshape((total_round, NROW, NCOL))
    bS_x = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_x').values()).reshape((total_round, NROW, NCOL))
    bS_y = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_y').values()).reshape((total_round, NROW, NCOL))
    bS_g = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_g').values()).reshape((total_round, NROW, NCOL))
    bS_w = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bS_w').values()).reshape((total_round, NROW, NCOL))
 
    # bulk add grey and white constriants
    for r in range(total_round):
        for i in ROW:
            for j in COL:             
                m.addConstr(fM_g[r,i,j] == gp.min_(fM_x[r,i,j], fM_y[r,i,j]))
                m.addConstr(fM_w[r,i,j] + fM_x[r,i,j] + fM_y[r,i,j] - fM_g[r,i,j] == 1)
                m.addConstr(bM_g[r,i,j] == gp.min_(bM_x[r,i,j], bM_y[r,i,j]))
                m.addConstr(bM_w[r,i,j] + bM_x[r,i,j] + bM_y[r,i,j] - bM_g[r,i,j] == 1)
                
                m.addConstr(fAL_g[r,i,j] == gp.min_(fAL_x[r,i,j], fAL_y[r,i,j]))
                m.addConstr(fAL_w[r,i,j] + fAL_x[r,i,j] + fAL_y[r,i,j] - fAL_g[r,i,j] == 1)
                m.addConstr(bAL_g[r,i,j] == gp.min_(bAL_x[r,i,j], bAL_y[r,i,j]))
                m.addConstr(bAL_w[r,i,j] + bAL_x[r,i,j] + bAL_y[r,i,j] - bAL_g[r,i,j] == 1)

                m.addConstr(fAR_g[r,i,j] == gp.min_(fAR_x[r,i,j], fAR_y[r,i,j]))
                m.addConstr(fAR_w[r,i,j] + fAR_x[r,i,j] + fAR_y[r,i,j] - fAR_g[r,i,j] == 1)
                m.addConstr(bAR_g[r,i,j] == gp.min_(bAR_x[r,i,j], bAR_y[r,i,j]))
                m.addConstr(bAR_w[r,i,j] + bAR_x[r,i,j] + bAR_y[r,i,j] - bAR_g[r,i,j] == 1)

                m.addConstr(fS_g[r,i,j] == gp.min_(fS_x[r,i,j], fS_y[r,i,j]))
                m.addConstr(fS_w[r,i,j] + fS_x[r,i,j] + fS_y[r,i,j] - fS_g[r,i,j] == 1)
                m.addConstr(bS_g[r,i,j] == gp.min_(bS_x[r,i,j], bS_y[r,i,j]))
                m.addConstr(bS_w[r,i,j] + bS_x[r,i,j] + bS_y[r,i,j] - bS_g[r,i,j] == 1) 

    # define GnD vars with constraints
    fAL_Gx = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAL_Gb').values()).reshape((total_round, NROW, NCOL))
    bAL_Gy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAL_Gr').values()).reshape((total_round, NROW, NCOL))
    fbAL_Gxy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='AL_Gbr').values()).reshape((total_round, NROW, NCOL))

    fAR_Gx = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='fAR_Gb').values()).reshape((total_round, NROW, NCOL))
    bAR_Gy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='bAR_Gr').values()).reshape((total_round, NROW, NCOL))
    fbAR_Gxy = np.asarray(m.addVars(total_round, NROW, NCOL, vtype= GRB.BINARY, name='AR_Gbr').values()).reshape((total_round, NROW, NCOL))
    
    # GnD rules (WLOG, fwd dir): we have in SupP w=0<=>x=1, w=1<=>x=0
    # when a cell is not white, it cannot be guessed (w=0 <=> x=1 -> Gx=1)
    # when a cell is white, it could be guessed (w=1 <=> x=0 ->Gx=0/1)
    # Hence, we have Gx>=x, whenever Gx>x, the bit is guessed, the cost of df in GnD will be: sum(Gx-x)
    # In GnD mode, Gx will be used in MC instead of x
    # if a cell is guessed in BiDr, then the cell in both dir must be white (fXw=bXw=1), and both are guessed to be non-zero value (Gx=Gy=1)
    for r in range(total_round):
        for i in ROW:
            for j in COL: 
                # structure: fML,bML <-[MC]-> fMR,bMR
                m.addConstr(fAL_Gx[r,i,j] >= fAL_x[r,i,j])
                m.addConstr(bAL_Gy[r,i,j] >= bAL_y[r,i,j])
                m.addConstr(fbAL_Gxy[r,i,j] == gp.min_(fAL_Gx[r,i,j], bAL_Gy[r,i,j], fAL_w[r,i,j], bAL_w[r,i,j]))

                m.addConstr(fAR_Gx[r,i,j] >= fAR_x[r,i,j])
                m.addConstr(bAR_Gy[r,i,j] >= bAR_y[r,i,j])
                m.addConstr(fbAR_Gxy[r,i,j] == gp.min_(fAR_Gx[r,i,j], bAR_Gy[r,i,j], fAR_w[r,i,j], bAR_w[r,i,j]))

    # define vars for columnwise encoding for MixCol input, including MC(fwd) and AK(bwd)
    fAL_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fAL_col_u').values()).reshape((total_round, NCOL))
    fAL_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fAL_col_v').values()).reshape((total_round, NCOL))
    fAL_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fAL_col_w').values()).reshape((total_round, NCOL))
    bAL_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bAL_col_u').values()).reshape((total_round, NCOL))
    bAL_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bAL_col_v').values()).reshape((total_round, NCOL))
    bAL_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bAL_col_w').values()).reshape((total_round, NCOL))
    fAR_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fAR_col_u').values()).reshape((total_round, NCOL))
    fAR_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fAR_col_v').values()).reshape((total_round, NCOL))
    fAR_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='fAR_col_w').values()).reshape((total_round, NCOL))
    bAR_col_u = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bAR_col_u').values()).reshape((total_round, NCOL))
    bAR_col_v = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bAR_col_v').values()).reshape((total_round, NCOL))
    bAR_col_w = np.asarray(m.addVars(total_round, NCOL, vtype=GRB.BINARY, name='bAR_col_w').values()).reshape((total_round, NCOL))

    # add constraints for u-v-w encoding
    for r in range(total_round):
        for j in COL:
            m.addConstr(fAL_col_v[r,j] == gp.min_(fAL_Gx[r,:,j].tolist()))
            m.addConstr(fAL_col_w[r,j] == gp.min_(fAL_y[r,:,j].tolist()))
            m.addConstr(fAL_col_u[r,j] == gp.max_(fAL_w[r,:,j].tolist()))
            m.addConstr(bAL_col_v[r,j] == gp.min_(bAL_x[r,:,j].tolist()))
            m.addConstr(bAL_col_w[r,j] == gp.min_(bAL_Gy[r,:,j].tolist()))
            m.addConstr(bAL_col_u[r,j] == gp.max_(bAL_w[r,:,j].tolist()))
            
            # calculated based on AR
            m.addConstr(fAR_col_v[r,j] == gp.min_(fAR_Gx[r,:,j].tolist()))
            m.addConstr(fAR_col_w[r,j] == gp.min_(fAR_y[r,:,j].tolist()))
            m.addConstr(fAR_col_u[r,j] == gp.max_(fAR_w[r,:,j].tolist()))
            m.addConstr(bAR_col_v[r,j] == gp.min_(bAR_x[r,:,j].tolist()))
            m.addConstr(bAR_col_w[r,j] == gp.min_(bAR_Gy[r,:,j].tolist()))
            m.addConstr(bAR_col_u[r,j] == gp.max_(bAR_w[r,:,j].tolist()))



    ### Key Vars (reminder, length = total_round + 1)
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
    
    # add grey and white indicators and constraints
    fK_g = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fK_g').values()).reshape((total_round + 1, NROW, NCOL))
    fK_w = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fK_w').values()).reshape((total_round + 1, NROW, NCOL))
    bK_g = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bK_g').values()).reshape((total_round + 1, NROW, NCOL))
    bK_w = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bK_w').values()).reshape((total_round + 1, NROW, NCOL)) 
    
    for r in range(total_round + 1):
        for i in ROW:
            for j in COL:             
                m.addConstr(fK_g[r,i,j] == gp.min_(fK_x[r,i,j], fK_y[r,i,j]))
                m.addConstr(fK_w[r,i,j] + fK_x[r,i,j] + fK_y[r,i,j] - fK_g[r,i,j] == 1)
                m.addConstr(bK_g[r,i,j] == gp.min_(bK_x[r,i,j], bK_y[r,i,j]))
                m.addConstr(bK_w[r,i,j] + bK_x[r,i,j] + bK_y[r,i,j] - bK_g[r,i,j] == 1)
    
    # add guess and determine bits and constriants
    fK_Gx = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='fK_Gb').values()).reshape((total_round + 1, NROW, NCOL))
    bK_Gy = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='bK_Gr').values()).reshape((total_round + 1, NROW, NCOL))
    fbK_Gxy = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype= GRB.BINARY, name='K_Gbr').values()).reshape((total_round + 1, NROW, NCOL))
    
    for r in range(total_round + 1):
        for i in ROW:
            for j in COL: 
                m.addConstr(fK_Gx[r,i,j] >= fK_x[r,i,j])
                m.addConstr(bK_Gy[r,i,j] >= bK_y[r,i,j])
                m.addConstr(fbK_Gxy[r,i,j] == gp.min_(fK_Gx[r,i,j], bK_Gy[r,i,j], fK_w[r,i,j], bK_w[r,i,j]))

    # add column-wise u-v-w encoding and constriants
    fK_col_u = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='fK_col_u').values()).reshape((total_round + 1, NCOL))
    fK_col_v = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='fK_col_v').values()).reshape((total_round + 1, NCOL))
    fK_col_w = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='fK_col_w').values()).reshape((total_round + 1, NCOL))
    bK_col_u = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='bK_col_u').values()).reshape((total_round + 1, NCOL))
    bK_col_v = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='bK_col_v').values()).reshape((total_round + 1, NCOL))
    bK_col_w = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='bK_col_w').values()).reshape((total_round + 1, NCOL))
    
    for r in range(total_round + 1):
        for j in COL:
            m.addConstr(fK_col_v[r,j] == gp.min_(fK_Gx[r,:,j].tolist()))
            m.addConstr(fK_col_w[r,j] == gp.min_(fK_y[r,:,j].tolist()))
            m.addConstr(fK_col_u[r,j] == gp.max_(fK_w[r,:,j].tolist()))
            m.addConstr(bK_col_v[r,j] == gp.min_(bK_x[r,:,j].tolist()))
            m.addConstr(bK_col_w[r,j] == gp.min_(bK_Gy[r,:,j].tolist()))
            m.addConstr(bK_col_u[r,j] == gp.max_(bK_w[r,:,j].tolist()))

    # add equivalent key for MulAK
    fK_invMC_x = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fK_invMC_x').values()).reshape((total_round + 1, NROW, NCOL))
    fK_invMC_y = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fK_invMC_y').values()).reshape((total_round + 1, NROW, NCOL))
    bK_invMC_x = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bK_invMC_x').values()).reshape((total_round + 1, NROW, NCOL))
    bK_invMC_y = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bK_invMC_y').values()).reshape((total_round + 1, NROW, NCOL))
    


    ### Cost Vars
    # define auxiliary vars tracking cost of df at MC operations
    mc_cost_fwd = np.asarray(m.addVars(total_round, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='MC_Cost_fwd').values()).reshape((total_round, NCOL))
    mc_cost_bwd = np.asarray(m.addVars(total_round, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='MC_Cost_bwd').values()).reshape((total_round, NCOL))
    
    # define auxiliary vars tracking cost of df for MulAK (moving the key before the MC layer)
    mc_inv_cost_fwd = np.asarray(m.addVars(total_round + 1, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='MC_INV_Cost_fwd').values()).reshape((total_round + 1, NCOL))
    mc_inv_cost_bwd = np.asarray(m.addVars(total_round + 1, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='MC_INV_Cost_bwd').values()).reshape((total_round + 1, NCOL))
    
    # define auxiliary vars tracking cost of df at Add Key operations in foward direction, at the LHS and the RHS of MC gate
    xor_lhs_cost_fwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='XOR_LHS_Cost_fwd').values()).reshape((total_round+1, NROW, NCOL))
    xor_lhs_cost_bwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='XOR_LHS_Cost_bwd').values()).reshape((total_round+1, NROW, NCOL))
    xor_rhs_cost_fwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='XOR_RHS_Cost_fwd').values()).reshape((total_round+1, NROW, NCOL))
    xor_rhs_cost_bwd = np.asarray(m.addVars(total_round+1, NROW, NCOL, vtype= GRB.BINARY, name='XOR_RHS_Cost_bwd').values()).reshape((total_round+1, NROW, NCOL))

    # define auxiliary vars trackin cost of df in the key expansion process, unpossible combinations are set to zeros
    key_cost_fwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((Nr, NROW, Nk))
    key_cost_bwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((Nr, NROW, Nk))
    


    ### Meet Vars
    # define final states for meet in the middle
    fMeet_lhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='fMeet_lhs_info').values()).reshape((NROW, NCOL))
    bMeet_lhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='bMeet_lhs_info').values()).reshape((NROW, NCOL))
    fMeet_rhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='fMeet_rhs_info').values()).reshape((NROW, NCOL))
    bMeet_rhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='bMeet_rhs_info').values()).reshape((NROW, NCOL))
    Meet_lhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_lhs_info').values()).reshape((NROW, NCOL))
    Meet_rhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_rhs_info').values()).reshape((NROW, NCOL))

    fTrace_lhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='fTrace_lhs_info').values()).reshape((NROW, NCOL))
    bTrace_lhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='bTrace_lhs_info').values()).reshape((NROW, NCOL))
    fTrace_rhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='fTrace_rhs_info').values()).reshape((NROW, NCOL))
    bTrace_rhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='bTrace_rhs_info').values()).reshape((NROW, NCOL))
    Trace_lhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Trace_lhs_info').values()).reshape((NROW, NCOL))
    Trace_rhs_info = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Trace_rhs_info').values()).reshape((NROW, NCOL))

    Meet_lhs_x = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_lhs_x').values()).reshape((NROW, NCOL))
    Meet_lhs_y = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_lhs_y').values()).reshape((NROW, NCOL))
    Meet_lhs_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_lhs_g').values()).reshape((NROW, NCOL)) 
    Meet_lhs_w = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_lhs_w').values()).reshape((NROW, NCOL)) 
    
    Meet_rhs_x = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_rhs_x').values()).reshape((NROW, NCOL))
    Meet_rhs_y = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_rhs_y').values()).reshape((NROW, NCOL))
    Meet_rhs_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_rhs_g').values()).reshape((NROW, NCOL)) 
    Meet_rhs_w = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Meet_rhs_w').values()).reshape((NROW, NCOL)) 
    
    # add constriants for grey indicators
    for i in ROW:
        for j in COL:
            m.addConstr(Meet_lhs_g[i,j] == gp.min_(Meet_lhs_x[i,j], Meet_lhs_y[i,j]))
            m.addConstr(Meet_lhs_w[i,j] + Meet_lhs_x[i,j] + Meet_lhs_y[i,j] - Meet_lhs_g[i,j] == 1)
            m.addConstr(Meet_rhs_g[i,j] == gp.min_(Meet_rhs_x[i,j], Meet_rhs_y[i,j]))
            m.addConstr(Meet_rhs_w[i,j] + Meet_rhs_x[i,j] + Meet_rhs_y[i,j] - Meet_rhs_g[i,j] == 1)
    
    # define auxiliary vars for computations on degree of matching
    meet = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Meet').values())



    ### MulAK
    # generate MC constraints on K and K_invMC, cost recorded in mc_inv
    for r in range(total_round + 1):
        for j in COL:
            gen_MC_rule(m, fK_Gx[r,:,j], fK_y[r,:,j], fK_col_u[r,j], fK_col_v[r,j], fK_col_w[r,j], fK_invMC_x[r,:,j], fK_invMC_y[r,:,j], mc_inv_cost_fwd[r,j], CONST0)
            gen_MC_rule(m, bK_x[r,:,j], bK_Gy[r,:,j], bK_col_u[r,j], bK_col_v[r,j], bK_col_w[r,j], bK_invMC_x[r,:,j], bK_invMC_y[r,:,j], CONST0, mc_inv_cost_bwd[r,j])

    # define MulAK indicators, used as conditional canstraints
    ind_MulAK = np.asarray(m.addVars(total_round + 1, NCOL, vtype=GRB.BINARY, name='MulAK_indicator').values()).reshape((total_round + 1, NCOL))
    
    # fix the MulAK be off at the last round add key process
    for j in COL:
        m.addConstr(ind_MulAK[total_round - 1, j] == 0)
        m.addConstr(ind_MulAK[total_round, j] == 0)

    # compose LHS and RHS equivalent key according to MulAK indicators
    # DEFAULT: KR
    fKL_x = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fKL_x').values()).reshape((total_round + 1, NROW, NCOL))
    fKL_y = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fKL_y').values()).reshape((total_round + 1, NROW, NCOL))
    bKL_x = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bKL_x').values()).reshape((total_round + 1, NROW, NCOL))
    bKL_y = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bKL_y').values()).reshape((total_round + 1, NROW, NCOL)) 
    
    fKR_x = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fKR_x').values()).reshape((total_round + 1, NROW, NCOL))
    fKR_y = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='fKR_y').values()).reshape((total_round + 1, NROW, NCOL))
    bKR_x = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bKR_x').values()).reshape((total_round + 1, NROW, NCOL))
    bKR_y = np.asarray(m.addVars(total_round + 1, NROW, NCOL, vtype=GRB.BINARY, name='bKR_y').values()).reshape((total_round + 1, NROW, NCOL)) 
    
    # perform column selection according to MulAK indicator
    for r in range(total_round + 1):
        for j in COL:
            for i in ROW:
                # MulAK==0: KR=K (the original key), KL==g (no influence)
                m.addConstr((ind_MulAK[r,j]==0) >> (fKR_x[r,i,j] == fK_x[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==0) >> (fKR_y[r,i,j] == fK_y[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==0) >> (bKR_x[r,i,j] == bK_x[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==0) >> (bKR_y[r,i,j] == bK_y[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==0) >> (fKL_x[r,i,j] == 1)) 
                m.addConstr((ind_MulAK[r,j]==0) >> (fKL_y[r,i,j] == 1))
                m.addConstr((ind_MulAK[r,j]==0) >> (bKL_x[r,i,j] == 1)) 
                m.addConstr((ind_MulAK[r,j]==0) >> (bKL_y[r,i,j] == 1))  

                # MulAK==1: KR==g (no influence), KL=K_invMC (the equivalent key after inv MC gate)
                m.addConstr((ind_MulAK[r,j]==1) >> (fKR_x[r,i,j] == 1)) 
                m.addConstr((ind_MulAK[r,j]==1) >> (fKR_y[r,i,j] == 1)) 
                m.addConstr((ind_MulAK[r,j]==1) >> (bKR_x[r,i,j] == 1)) 
                m.addConstr((ind_MulAK[r,j]==1) >> (bKR_y[r,i,j] == 1)) 
                m.addConstr((ind_MulAK[r,j]==1) >> (fKL_x[r,i,j] == fK_invMC_x[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==1) >> (fKL_y[r,i,j] == fK_invMC_y[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==1) >> (bKL_x[r,i,j] == bK_invMC_x[r,i,j])) 
                m.addConstr((ind_MulAK[r,j]==1) >> (bKL_y[r,i,j] == bK_invMC_y[r,i,j])) 



    ### Obj vars
    df_b = m.addVar(lb=3, ub=5, vtype=GRB.INTEGER, name="DF_b")
    df_r = m.addVar(lb=3, ub=5, vtype=GRB.INTEGER, name="DF_r")
    dm = m.addVar(lb=3, ub=5, vtype=GRB.INTEGER, name="Match")
    obj = m.addVar(lb=3, ub=5, vtype=GRB.INTEGER, name="Obj")

    GnD_b = m.addVar(lb=0, vtype=GRB.INTEGER, name="GND_b")
    GnD_r = m.addVar(lb=0, vtype=GRB.INTEGER, name="GND_r")
    GnD_br = m.addVar(lb=0, vtype=GRB.INTEGER, name="GND_br")

    m.addConstr(GnD_b == gp.quicksum(fAL_Gx.flatten()) - gp.quicksum(fAL_x.flatten()) + gp.quicksum(fAR_Gx.flatten()) - gp.quicksum(fAR_x.flatten()) + gp.quicksum(fK_Gx.flatten()) - gp.quicksum(fK_x.flatten()))
    m.addConstr(GnD_r == gp.quicksum(bAL_Gy.flatten()) - gp.quicksum(bAL_y.flatten()) + gp.quicksum(bAR_Gy.flatten()) - gp.quicksum(bAR_y.flatten()) + gp.quicksum(bK_Gy.flatten()) - gp.quicksum(bK_y.flatten())) 
    m.addConstr(GnD_br == gp.quicksum(fbAL_Gxy.flatten()) + gp.quicksum(fbAR_Gxy.flatten()) + gp.quicksum(fbK_Gxy.flatten()))




#### Tweak Panel ####
    # MulAK switch: default off
    for r in range(total_round - 1):
        for j in COL:
            #continue
            if r == (match_round-1)%total_round or r == (match_round+1)%total_round:
                pass
            else: 
                m.addConstr(ind_MulAK[r,j]==0)
    
    # BiDir switch: default on
    for r in range(total_round + 1):
        if r == match_round:
            pass
    
    # GnD switch: default off
    # default GnD off
    #m.addConstr(GnD_b == 0)
    #m.addConstr(GnD_r == 0)
    #m.addConstr(GnD_br == 0)

    # proposition: fix start state to be all red (WLOG), in compensate of the efficiency
    for i in ROW:
        for j in COL:
            continue
            m.addConstr(E_ini_y[i, j] == 1) #test
            m.addConstr(E_ini_x[i, j] == 0) #test

    # possible optimal for key start at 4
    for i in ROW:
        for j in range(Nk):
            continue
            if i==j:
                m.addConstr(E_ini_y[i, j] == 0) #test
                m.addConstr(E_ini_x[i, j] == 1) #test
            else: 
                m.addConstr(E_ini_y[i, j] == 1) #test
                m.addConstr(E_ini_x[i, j] == 0) #test

    # possible optimal for key start at 3
    for i in ROW:
        for j in range(Nk):
            continue
            if (i==0 and j==2) or (i==2 and j==2):
                m.addConstr(K_ini_y[i, j] == 0) #test
                m.addConstr(K_ini_x[i, j] == 1) #test
            else: 
                m.addConstr(K_ini_y[i, j] == 1) #test
                m.addConstr(K_ini_x[i, j] == 0) #test

    # test for key schedule
    for i in ROW:
        for j in range(Nk):
            continue
            if (i==1 and j==2) or (i==3 and j==2) or (i==1 and j==3) or (i==3 and j==3):
                m.addConstr(K_ini_y[i, j] == 0) #test
                m.addConstr(K_ini_x[i, j] == 1) #test
            else: 
                m.addConstr(K_ini_y[i, j] == 1) #test
                m.addConstr(K_ini_x[i, j] == 0) #test
    
    for i in ROW:
        for j in COL:
            continue
            if j < 2:
                m.addConstr(M_y[i, j] == 1) #test
                m.addConstr(M_x[i, j] == 0) #test
            else: 
                m.addConstr(M_y[i, j] == 0) #test
                m.addConstr(M_x[i, j] == 1) #test



#### Main Procedure ####
    print(m.modelName)
    print(Nb, Nk, Nr)

    # add constriants according to the key expansion algorithm
    key_expansion(m, key_size, total_round, key_start_round, K_ini_x, K_ini_y, fKS_x, fKS_y, fKS_g, fKS_c, fKS_eq_g, bKS_x, bKS_y, bKS_g, bKS_c, bKS_eq_g, CONST0, key_cost_fwd, key_cost_bwd)
    
    # take into account of the key relation and adjust the true key expansion cost
    true_key_cost_fwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='true_key_cost_fwd').values()).reshape((Nr, NROW, Nk))
    true_key_cost_bwd = np.asarray(m.addVars(Nr, NROW, Nk, vtype= GRB.BINARY, name='true_key_cost_bwd').values()).reshape((Nr, NROW, Nk))
    for r in range(Nr):
        for i in range(Nb):
            for j in range(Nk):
                m.addConstr(true_key_cost_fwd[r,i,j] == key_cost_fwd[r,i,j] - fKS_eq_g[r,i,j])
                m.addConstr(true_key_cost_bwd[r,i,j] == key_cost_bwd[r,i,j] - bKS_eq_g[r,i,j])

    # initialize the enc states, avoid unknown to maximize performance
    for i in ROW:
        for j in COL:
            m.addConstr(S_x[enc_start_round, i, j] + S_y[enc_start_round, i, j] >= 1)
            m.addConstr(E_ini_x[i, j] == S_x[enc_start_round, i, j])
            m.addConstr(E_ini_y[i, j] == S_y[enc_start_round, i, j])

    # add constriants according to the encryption algorithm
    for r in range(total_round):
        nr = r + 1   # alias for next round
        
        # special matching: identity meet at last round, only allows same pure colored cells to match 
        # DEFAULT: using xor_rhs, AK_RHS and K_RHS (same as K in this situation), and fix xor_lhs == 0
        if r == match_round and match_round == total_round - 1:
            print('mat lastr', r)
            for i in ROW:
                for j in COL:
                    # Enter SupP at last round MC state
                    ent_SupP(m, M_x[r,i,j], M_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j])
                    # add last round key: AK_RHS[lr](storing AT) = id(MC[lr]) XOR KEY[lr]
                    gen_XOR_rule(m, fM_x[r,i,j], fM_y[r,i,j], fKR_x[r,i,j], fKR_y[r,i,j], fAR_x[r,i,j], fAR_y[r,i,j], xor_rhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bM_x[r,i,j], bM_y[r,i,j], bKR_x[r,i,j], bKR_y[r,i,j], bAR_x[r,i,j], bAR_y[r,i,j], CONST0, xor_rhs_cost_bwd[r,i,j])
                    # add whitening key: SB[0] = AK[lr](storing AT) XOR KEY[tr](storing KEY[-1])
                    gen_XOR_rule(m, fAR_x[r,i,j], fAR_y[r,i,j], fKR_x[nr,i,j], fKR_y[nr,i,j], fS_x[0,i,j], fS_y[0,i,j], xor_rhs_cost_fwd[nr,i,j], CONST0)
                    gen_XOR_rule(m, bAR_x[r,i,j], bAR_y[r,i,j], bKR_x[nr,i,j], bKR_y[nr,i,j], bS_x[0,i,j], bS_y[0,i,j], CONST0, xor_rhs_cost_bwd[nr,i,j])
                    # Exit SupP at Meet_fwd
                    ext_SupP(m, fS_x[0,i,j], fS_y[0,i,j], bS_x[0,i,j], bS_y[0,i,j], Meet_lhs_x[i,j], Meet_lhs_y[i,j]) 
                    # Fix xor_lhs == 0
                    m.addConstr(xor_lhs_cost_fwd[r,i,j] == 0) 
                    m.addConstr(xor_lhs_cost_bwd[r,i,j] == 0) 
                    m.addConstr(xor_lhs_cost_fwd[nr,i,j] == 0) 
                    m.addConstr(xor_lhs_cost_bwd[nr,i,j] == 0) 
            # calculate degree of match
            tempMeet = np.asarray(m.addVars(NROW, NCOL, vtype= GRB.BINARY, name='tempMeet').values()).reshape((NROW, NCOL))
            for j in COL:
                for i in ROW:
                    m.addConstr(tempMeet[i,j] == gp.max_(Meet_lhs_w[i,j], S_w[0,i,j]))
                m.addConstr(meet[j] == NROW - gp.quicksum(tempMeet[:,j]))
            continue
        
        # normal matching
        elif r == match_round:
            print('mat', r)  
            lr = (r - 1 + total_round) % total_round
            for i in ROW:
                for j in COL:
                    #continue
                    # RHS of the MC
                    # Enter SupP at next round SB state, to SupP NSB
                    ent_SupP(m, S_x[nr,i,j], S_y[nr,i,j], fS_x[nr,i,j], fS_y[nr,i,j], bS_x[nr,i,j], bS_y[nr,i,j])
                    # (reverse) Add eq rhs key with SupP to NSB, get AK_RHS  
                    gen_XOR_rule(m, fS_x[nr,i,j], fS_y[nr,i,j], fKR_x[r,i,j], fKR_y[r,i,j], fAR_x[r,i,j], fAR_y[r,i,j], xor_rhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bS_x[nr,i,j], bS_y[nr,i,j], bKR_x[r,i,j], bKR_y[r,i,j], bAR_x[r,i,j], bAR_y[r,i,j], CONST0, xor_rhs_cost_bwd[r,i,j])
                    # exit SupP at rhs of MC gate (for 4 same color matching)
                    ext_SupP(m, fAR_x[r,i,j], fAR_y[r,i,j], bAR_x[r,i,j], bAR_y[r,i,j], Meet_rhs_x[i,j], Meet_rhs_y[i,j])

                    # LHS of the MC
                    #Meet_lhs_x, Meet_lhs_y, Meet_lhs_g = M_x[r,:,:], M_y[r,:,:], M_g[r,:,:]
                    # Enter SupP at current round MC state, to SupP AK_LHS
                    ent_SupP(m, M_x[r,i,j], M_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j])
                    # Add eq lhs key with SupP to AK_LHS, get MC  
                    gen_XOR_rule(m, fM_x[r,i,j], fM_y[r,i,j], fKL_x[r,i,j], fKL_y[r,i,j], fAL_x[r,i,j], fAL_y[r,i,j], xor_lhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bM_x[r,i,j], bM_y[r,i,j], bKL_x[r,i,j], bKL_y[r,i,j], bAL_x[r,i,j], bAL_y[r,i,j], CONST0, xor_lhs_cost_bwd[r,i,j])
                    # exit SupP at lhs of MC gate (for 4 same color matching)
                    ext_SupP(m, fAL_x[r,i,j], fAL_y[r,i,j], bAL_x[r,i,j], bAL_y[r,i,j], Meet_lhs_x[i,j], Meet_lhs_y[i,j])
                    
                    # Match in SupP state: determine if both of the superposition branches carry information (i.e. non-white) 
                    # the info var marks if the meet state contains information thus could be used for matching
                    m.addConstr(fMeet_rhs_info[i,j] == 1 - fAR_w[r,i,j]) 
                    m.addConstr(bMeet_rhs_info[i,j] == 1 - bAR_w[r,i,j])
                    m.addConstr(Meet_rhs_info[i,j] == gp.min_(fMeet_rhs_info[i,j], bMeet_rhs_info[i,j]))
                    
                    m.addConstr(fMeet_lhs_info[i,j] == 1 - fAL_w[r,i,j]) 
                    m.addConstr(bMeet_lhs_info[i,j] == 1 - bAL_w[r,i,j])
                    m.addConstr(Meet_lhs_info[i,j] == gp.min_(fMeet_lhs_info[i,j], bMeet_lhs_info[i,j]))

                    m.addConstr(fTrace_rhs_info[i,j] == 1 - fM_w[r+1,i,(j-i+NCOL)%NCOL]) 
                    m.addConstr(bTrace_rhs_info[i,j] == 1 - bM_w[r+1,i,(j-i+NCOL)%NCOL]) 
                    m.addConstr(Trace_rhs_info[i,j] == gp.min_(fTrace_rhs_info[i,j], fKR_x[r,i,j], bTrace_rhs_info[i,j], bKR_y[r,i,j])) 

                    m.addConstr(fTrace_lhs_info[i,j] == 1 - fS_w[r,i,(j+i+NCOL)%NCOL]) 
                    m.addConstr(bTrace_lhs_info[i,j] == 1 - bS_w[r,i,(j+i+NCOL)%NCOL]) 
                    m.addConstr(Trace_lhs_info[i,j] == gp.min_(fTrace_lhs_info[i,j], fKL_x[r,i,j], bTrace_lhs_info[i,j], bKL_y[r,i,j])) 
            
            # generate match rule
            #gen_match_rule(m, Meet_lhs_x, Meet_lhs_y, Meet_lhs_g, Meet_lhs_info, Meet_rhs_x, Meet_rhs_y, Meet_rhs_g, Meet_rhs_info, meet)
            #gen_new_match_rule(m, Meet_lhs_x, Meet_lhs_y, Meet_lhs_info, Meet_rhs_x, Meet_rhs_y, Meet_rhs_info, meet)
            gen_combined_match_rule(m, Meet_lhs_x, Meet_lhs_y, Meet_lhs_info, Trace_lhs_info, Meet_rhs_x, Meet_rhs_y, Meet_rhs_info, Trace_rhs_info, meet)
            continue
        
        # last round
        elif r == total_round - 1:
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
                        gen_XOR_rule(m, fM_x[r,i,j], fM_y[r,i,j], fKR_x[r,i,j], fKR_y[r,i,j], fAR_x[r,i,j], fAR_y[r,i,j], xor_rhs_cost_fwd[r,i,j], CONST0)
                        gen_XOR_rule(m, bM_x[r,i,j], bM_y[r,i,j], bKR_x[r,i,j], bKR_y[r,i,j], bAR_x[r,i,j], bAR_y[r,i,j], CONST0, xor_rhs_cost_bwd[r,i,j])
                        # add whitening key: SB[0] = AK[lr](storing AT) XOR KEY[tr](storing KEY[-1])
                        gen_XOR_rule(m, fAR_x[r,i,j], fAR_y[r,i,j], fKR_x[nr,i,j], fKR_y[nr,i,j], fS_x[0,i,j], fS_y[0,i,j], xor_rhs_cost_fwd[nr,i,j], CONST0)
                        gen_XOR_rule(m, bAR_x[r,i,j], bAR_y[r,i,j], bKR_x[nr,i,j], bKR_y[nr,i,j], bS_x[0,i,j], bS_y[0,i,j], CONST0, xor_rhs_cost_bwd[nr,i,j])
                        # Exit SupP at round 0 SB state
                        ext_SupP(m, fS_x[0,i,j], fS_y[0,i,j], bS_x[0,i,j], bS_y[0,i,j], S_x[0,i,j], S_y[0,i,j])  
            elif r in bwd:  # enter last round in bwd direction
                for i in ROW:
                    for j in COL:
                        # Enter SupP at round 0 SB state
                        ent_SupP(m, S_x[0,i,j], S_y[0,i,j], fS_x[0,i,j], fS_y[0,i,j], bS_x[0,i,j], bS_y[0,i,j])
                        # add whitening key: AK[tr-1](storing AT) =  SB[0] XOR KEY[tr](storing KEY[-1])
                        gen_XOR_rule(m, fS_x[0,i,j], fS_y[0,i,j], fKR_x[nr,i,j], fKR_y[nr,i,j], fAR_x[r,i,j], fAR_y[r,i,j], xor_rhs_cost_fwd[nr,i,j], CONST0)
                        gen_XOR_rule(m, bS_x[0,i,j], bS_y[0,i,j], bKR_x[nr,i,j], bKR_y[nr,i,j], bAR_x[r,i,j], bAR_y[r,i,j], CONST0, xor_rhs_cost_bwd[nr,i,j])
                        # add last round key: MC[lr] == id(MC[lr]) = AK[lr](storing AT) XOR KEY[lr]
                        gen_XOR_rule(m, fAR_x[r,i,j], fAR_y[r,i,j], fKR_x[r,i,j], fKR_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], xor_rhs_cost_fwd[r,i,j], CONST0)
                        gen_XOR_rule(m, bAR_x[r,i,j], bAR_y[r,i,j], bKR_x[r,i,j], bKR_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], CONST0, xor_rhs_cost_bwd[r,i,j])
                        # Ext SupP feed the outcome to current MC state
                        ext_SupP(m, fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], M_x[r,i,j], M_y[r,i,j])
            # fix xor_lhs cost
            for i in ROW:
                for j in COL:
                    m.addConstr(xor_lhs_cost_fwd[nr,i,j] == 0)
                    m.addConstr(xor_lhs_cost_bwd[nr,i,j] == 0)
                    m.addConstr(xor_lhs_cost_fwd[r,i,j] == 0)
                    m.addConstr(xor_lhs_cost_bwd[r,i,j] == 0)

        
        # ENC Propagation: forward direction
        elif r in fwd:
            #continue
            # Enter SupP at current round AK_LHS
            print('fwd', r)
            for i in ROW:
                for j in COL:
                    ent_SupP(m, M_x[r,i,j], M_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j])
            # Add eq LHS Key with SupP to AK_LHS, get MC  
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fM_x[r,i,j], fM_y[r,i,j], fKL_x[r,i,j], fKL_y[r,i,j], fAL_x[r,i,j], fAL_y[r,i,j], xor_lhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bM_x[r,i,j], bM_y[r,i,j], bKL_x[r,i,j], bKL_y[r,i,j], bAL_x[r,i,j], bAL_y[r,i,j], CONST0, xor_lhs_cost_bwd[r,i,j])
            # MixCol with SupP and GnD, between MC and AK_RHS
            for j in COL:
                #continue
                gen_MC_rule(m, fAL_Gx[r,:,j], fAL_y[r,:,j], fAL_col_u[r,j], fAL_col_v[r,j], fAL_col_w[r,j], fAR_x[r,:,j], fAR_y[r,:,j], mc_cost_fwd[r,j], CONST0)
                gen_MC_rule(m, bAL_x[r,:,j], bAL_Gy[r,:,j], bAL_col_u[r,j], bAL_col_v[r,j], bAL_col_w[r,j], bAR_x[r,:,j], bAR_y[r,:,j], CONST0, mc_cost_bwd[r,j])
                for i in ROW:   # fix unused guess bits
                    m.addConstr(fAR_x[r,i,j] == fAR_Gx[r,i,j])
                    m.addConstr(bAR_y[r,i,j] == bAR_Gy[r,i,j])
            # Add eq RHS Key with SupP to AK_RHS, get NSB    
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fAR_x[r,i,j], fAR_y[r,i,j], fKR_x[r,i,j], fKR_y[r,i,j], fS_x[nr,i,j], fS_y[nr,i,j], xor_rhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bAR_x[r,i,j], bAR_y[r,i,j], bKR_x[r,i,j], bKR_y[r,i,j], bS_x[nr,i,j], bS_y[nr,i,j], CONST0, xor_rhs_cost_bwd[r,i,j])
            # Ext SupP and feed the outcome to next SB state
            for i in ROW:
                for j in COL:
                    ext_SupP(m, fS_x[nr,i,j], fS_y[nr,i,j], bS_x[nr,i,j], bS_y[nr,i,j], S_x[nr,i,j], S_y[nr,i,j])
            continue
        
        # ENC Propagation: backward direction
        elif r in bwd:
            #continue
            print('bwd', r)
            # Enter SupP at next round SB state
            for i in ROW:
                for j in COL:
                    ent_SupP(m, S_x[nr,i,j], S_y[nr,i,j], fS_x[nr,i,j], fS_y[nr,i,j], bS_x[nr,i,j], bS_y[nr,i,j])
            # (reverse) Add eq RHS key with SupP to NSB, get AK_RHS    
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fS_x[nr,i,j], fS_y[nr,i,j], fKR_x[r,i,j], fKR_y[r,i,j], fAR_x[r,i,j], fAR_y[r,i,j], xor_rhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bS_x[nr,i,j], bS_y[nr,i,j], bKR_x[r,i,j], bKR_y[r,i,j], bAR_x[r,i,j], bAR_y[r,i,j], CONST0, xor_rhs_cost_bwd[r,i,j])
            # (reverse) MixCol with SupP and GnD, between AK_RHS and MC
            for j in COL:
                #continue
                gen_MC_rule(m, fAR_Gx[r,:,j], fAR_y[r,:,j], fAR_col_u[r,j], fAR_col_v[r,j], fAR_col_w[r,j], fAL_x[r,:,j], fAL_y[r,:,j], mc_cost_fwd[r,j], CONST0)
                gen_MC_rule(m, bAR_x[r,:,j], bAR_Gy[r,:,j], bAR_col_u[r,j], bAR_col_v[r,j], bAR_col_w[r,j], bAL_x[r,:,j], bAL_y[r,:,j], CONST0, mc_cost_bwd[r,j])
                for i in ROW:   # fix unused guess bits
                    m.addConstr(fAL_x[r,i,j] == fAL_Gx[r,i,j])
                    m.addConstr(bAL_y[r,i,j] == bAL_Gy[r,i,j])
            # (reverse) Add eq LHS key with SupP to MC, get AK_LHS  
            for i in ROW:
                for j in COL:
                    #continue
                    gen_XOR_rule(m, fAL_x[r,i,j], fAL_y[r,i,j], fKL_x[r,i,j], fKL_y[r,i,j], fM_x[r,i,j], fM_y[r,i,j], xor_lhs_cost_fwd[r,i,j], CONST0)
                    gen_XOR_rule(m, bAL_x[r,i,j], bAL_y[r,i,j], bKL_x[r,i,j], bKL_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], CONST0, xor_lhs_cost_bwd[r,i,j])
            # Ext SupP and feed the outcome to current MC state
            for i in ROW:
                for j in COL:
                    ext_SupP(m, fM_x[r,i,j], fM_y[r,i,j], bM_x[r,i,j], bM_y[r,i,j], M_x[r,i,j], M_y[r,i,j])
            continue
        else:
            raise Exception("Irregular Behavior at encryption")
    
    # set objective function
    m.addConstr(df_b == gp.quicksum(E_ini_x.flatten()) - gp.quicksum(E_ini_g.flatten()) + gp.quicksum(K_ini_x.flatten()) - gp.quicksum(K_ini_g.flatten()) - gp.quicksum(mc_cost_fwd.flatten()) - gp.quicksum(mc_inv_cost_fwd.flatten()) - gp.quicksum(xor_lhs_cost_fwd.flatten()) - gp.quicksum(xor_rhs_cost_fwd.flatten()) - gp.quicksum(true_key_cost_fwd.flatten()) )
    m.addConstr(df_r == gp.quicksum(E_ini_y.flatten()) - gp.quicksum(E_ini_g.flatten()) + gp.quicksum(K_ini_y.flatten()) - gp.quicksum(K_ini_g.flatten()) - gp.quicksum(mc_cost_bwd.flatten()) - gp.quicksum(mc_inv_cost_bwd.flatten()) - gp.quicksum(xor_lhs_cost_bwd.flatten()) - gp.quicksum(xor_rhs_cost_bwd.flatten()) - gp.quicksum(true_key_cost_bwd.flatten()) )
    m.addConstr(dm == gp.quicksum(meet.flatten()))
   
    m.addConstr(obj <= df_b - GnD_r)
    m.addConstr(obj <= df_r - GnD_b)
    m.addConstr(obj <= dm - GnD_b - GnD_r - GnD_br)
    m.setObjective(obj, GRB.MAXIMIZE)

    # set parameters
    m.setParam(GRB.Param.PoolSearchMode, 2)
    m.setParam(GRB.Param.PoolSolutions,  1)
    #m.setParam(GRB.Param.BestObjStop, 2.999999999)
    m.setParam(GRB.Param.Threads, 8)

    # optimization
    start_time = time.time()
    m.optimize()
    end_time = time.time()
    time_cost = end_time - start_time
    
    if not os.path.exists(path = dir):
        os.makedirs(dir)
    
    m.write(dir + m.modelName + '.lp')

    if m.SolCount > 0:
        for sol_i in range(m.SolCount):
            m.write(dir + m.modelName + '_sol_' + str(sol_i) + '.sol')
            tex_display(key_size, total_round, enc_start_round, match_round, key_start_round, m.modelName, sol_i, time_cost, dir)
        return m.SolCount
    else:
        return 0

#solve(key_size=192, total_round=8, enc_start_round=3, match_round=6, key_start_round=3, dir='./AES_SupP_GnD_RKc_NewMatch_MulAK/runs/')

# batch search
for r in range(9):
    continue
    if r != 4:
        print('AES192-9%d43 Fix Key Start\n' %r)
        solve(key_size=192, total_round=9, enc_start_round=r, match_round=4, key_start_round=3, dir='./AES_SupP_GnD_RKc_NewMatch_MulAK/192_9X43_fixKey/')

solve(key_size=192, total_round=8, enc_start_round=3, match_round=6, key_start_round=3, dir='./AES_SupP_GnD_RKc_NewMatch_MulAK/8r_runs/')