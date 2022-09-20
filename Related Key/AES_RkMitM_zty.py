from io import TextIOWrapper
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import re
import os

# AES parameters
NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)
TAB = ' '*4

#m = gp.Model('%dR_ENC_r%d_KEY_r%dMeet_r%d' % (total_round, start_round, key_start_round, match_round))

def def_var(m:gp.Model, total_r: int, fwd, bwd):
    # define vars storing the SB state at each round with encoding scheme
    S_b = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='S_b').values()).reshape((total_r, NROW, NCOL))
    S_r = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='S_r').values()).reshape((total_r, NROW, NCOL))
    S_g = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='S_g').values()).reshape((total_r, NROW, NCOL))
    S_w = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='S_w').values()).reshape((total_r, NROW, NCOL))

    # define alias storing the MC state at each round with encoding scheme
    M_b = np.ndarray(shape= (total_r, NROW, NCOL), dtype= gp.Var)
    M_r = np.ndarray(shape= (total_r, NROW, NCOL), dtype= gp.Var)
    M_g = np.ndarray(shape= (total_r, NROW, NCOL), dtype= gp.Var)
    M_w = np.ndarray(shape= (total_r, NROW, NCOL), dtype= gp.Var)
    
    # match the cells with alias through shift rows
    for r in range(total_r):
        for i in ROW:
            for j in COL:   
                M_b[r,i,j] = S_b[r,i,(j+i)%NCOL]
                M_r[r,i,j] = S_r[r,i,(j+i)%NCOL]
                M_g[r,i,j] = S_g[r,i,(j+i)%NCOL]
                M_w[r,i,j] = S_w[r,i,(j+i)%NCOL]

    # define vars storing the Add key state at each round with encoding scheme
    A_b = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='A_b').values()).reshape((total_r, NROW, NCOL))
    A_r = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='A_r').values()).reshape((total_r, NROW, NCOL))
    A_g = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='A_g').values()).reshape((total_r, NROW, NCOL))
    A_w = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='A_w').values()).reshape((total_r, NROW, NCOL))
    
    # define vars storing the key state at each round with encoding scheme, add 1 more entry to store the whitening key
    K_b = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='K_b').values()).reshape((total_r+1, NROW, NCOL))
    K_r = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='K_r').values()).reshape((total_r+1, NROW, NCOL))
    K_g = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='K_g').values()).reshape((total_r+1, NROW, NCOL))
    K_w = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='K_w').values()).reshape((total_r+1, NROW, NCOL))

    # define vars for columnwise encoding, including SB, MC and Key
    S_col_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_u').values()).reshape((total_r, NCOL))
    S_col_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_x').values()).reshape((total_r, NCOL))
    S_col_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_y').values()).reshape((total_r, NCOL))

    M_col_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_u').values()).reshape((total_r, NCOL))
    M_col_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_x').values()).reshape((total_r, NCOL))
    M_col_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_y').values()).reshape((total_r, NCOL))

    K_col_u = np.asarray(m.addVars(total_r+1, NCOL, vtype=GRB.BINARY, name='K_col_u').values()).reshape((total_r+1, NCOL))
    K_col_x = np.asarray(m.addVars(total_r+1, NCOL, vtype=GRB.BINARY, name='K_col_x').values()).reshape((total_r+1, NCOL))
    K_col_y = np.asarray(m.addVars(total_r+1, NCOL, vtype=GRB.BINARY, name='K_col_y').values()).reshape((total_r+1, NCOL))

    # define vars for XorMC encodings
    XorMC_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='XORMC_u').values()).reshape((total_r, NCOL))
    XorMC_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='XORMC_x').values()).reshape((total_r, NCOL))
    XorMC_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='XORMC_y').values()).reshape((total_r, NCOL))
    XorMC_z = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='XORMC_z').values()).reshape((total_r, NCOL))
    XorMC = np.asarray(m.addVars(total_r, NROW, NCOL, vtype=GRB.BINARY, name='XORMC').values()).reshape((total_r, NROW, NCOL))

    # define vars to track the start state of Encryption states (S) and Key states (K)
    S_ini_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='S_ini_b').values()).reshape((NROW, NCOL))
    S_ini_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='S_ini_r').values()).reshape((NROW, NCOL))
    S_ini_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='S_ini_g').values()).reshape((NROW, NCOL))   
    K_ini_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_b').values()).reshape((NROW, NCOL))
    K_ini_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_r').values()).reshape((NROW, NCOL))
    K_ini_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_g').values()).reshape((NROW, NCOL))
    
    # define auxiliary vars tracking cost of df at MC operations, cost_fwd is solely for MC, cost_bwd is for XOR_MC
    cost_fwd = np.asarray(m.addVars(total_r, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Cost_fwd').values()).reshape((total_r, NCOL))
    cost_bwd = np.asarray(m.addVars(total_r, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Cost_bwd').values()).reshape((total_r, NCOL))
    
    # define auxiliary vars tracking cost of df at Add Key operations in foward direction
    cost_XOR = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='Cost_XOR').values()).reshape((total_r+1, NROW, NCOL))

    # define auxiliary vars trackin cost of df in the key expansion process, unpossible combinations are set to zeros
    key_cost_fwd = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((total_r+1, NROW, NCOL))
    key_cost_bwd = np.asarray(m.addVars(total_r+1, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((total_r+1, NROW, NCOL))
    
    # define auxiliary vars for computations on degree of matching
    meet_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype=GRB.INTEGER, name='Meet_signed').values())
    meet = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Meet').values())
    
    # add encoding constraints for key and internal states
    for r in range(total_r):
        for i in ROW:
            for j in COL:
                m.addConstr(S_g[r,i,j] == gp.and_(S_b[r,i,j], S_r[r,i,j]))
                m.addConstr(S_w[r,i,j] + S_b[r,i,j] + S_r[r,i,j] - S_g[r,i,j] == 1)
                
                m.addConstr(A_g[r,i,j] == gp.and_(A_b[r,i,j], A_r[r,i,j]))
                m.addConstr(A_w[r,i,j] + A_b[r,i,j] + A_r[r,i,j] - A_g[r,i,j] == 1)

    for r in range(total_r+1):
        for i in ROW:
            for j in COL:
                m.addConstr(K_g[r,i,j] == gp.and_(K_b[r,i,j], K_r[r,i,j]))
                m.addConstr(K_w[r,i,j] + K_b[r,i,j] + K_r[r,i,j] - K_g[r,i,j] == 1)

    for r in range(total_r):
        for j in COL:
            m.addConstr(S_col_x[r,j] == gp.min_(S_b[r,:,j].tolist()))
            m.addConstr(S_col_y[r,j] == gp.min_(S_r[r,:,j].tolist()))
            m.addConstr(S_col_u[r,j] == gp.max_(S_w[r,:,j].tolist()))

            m.addConstr(M_col_x[r,j] == gp.min_(M_b[r,:,j].tolist()))
            m.addConstr(M_col_y[r,j] == gp.min_(M_r[r,:,j].tolist()))
            m.addConstr(M_col_u[r,j] == gp.max_(M_w[r,:,j].tolist()))
    
    for r in range(total_r+1):
        for j in COL:
            m.addConstr(K_col_x[r,j] == gp.min_(K_b[r,:,j].tolist()))
            m.addConstr(K_col_y[r,j] == gp.min_(K_r[r,:,j].tolist()))
            m.addConstr(K_col_u[r,j] == gp.max_(K_w[r,:,j].tolist()))
    
    TAU_A = np.asarray([
        [0,0,0,0,1],[-1,0,0,0,0],[0,-1,0,0,-1],[0,0,-1,0,0],[0,0,0,-1,-1],[0,0,0,1,0],[0,0,1,0,-1],[0,1,0,0,0],[1,0,0,0,-1],[-1,1,-1,1,1]
    ])
    TAU_B = np.asarray([0,1,1,1,1,0,0,0,0,1])

    for r in bwd:
        nr = (r+1) % total_r
        for j in COL:
            m.addConstr(XorMC_u[r,j] == gp.or_(S_col_u[nr,j], K_col_u[r,j]))
            m.addConstr(XorMC_x[r,j] == gp.and_(S_col_x[nr,j], K_col_x[r,j]))
            m.addConstr(XorMC_y[r,j] == gp.and_(S_col_y[nr,j], K_col_y[r,j]))
            m.addConstr(XorMC_z[r,j] == gp.min_(S_r[nr,:,j].tolist() + K_r[r,:,j].tolist()))
            for i in ROW:
                #m.addMConstr(TAU_A, (S_b[nr,i,j], S_r[nr,i,j], K_b[r,i,j], K_r[r,i,j], XorMC[r,i,j]), '>=', -TAU_B)
                m.addConstr(XorMC[r,i,j] == gp.or_(S_b[nr,i,j], K_b[r,i,j]))

    m.update()
    return [
        S_b, S_r, S_g, S_w, M_b, M_r, M_g, M_w, A_b, A_r, A_g, A_w, K_b, K_r, K_g, K_w,
        S_col_u, S_col_x, S_col_y, M_col_u, M_col_x, M_col_y, K_col_u, K_col_x, K_col_y,
        XorMC, XorMC_u, XorMC_x, XorMC_y, XorMC_z,
        S_ini_b, S_ini_r, S_ini_g, K_ini_b, K_ini_r, K_ini_g,
        cost_fwd, cost_XOR, cost_bwd, key_cost_fwd, key_cost_bwd, meet_signed, meet]

# generate XOR rule for forward computations, if backward, switch the input of blue and red
def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_df: gp.Var):
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

    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df]
    m.addMConstr(XOR_A, list(enum), '>=', -XOR_B)

# generate MC rule 
def gen_MC_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_col_u: gp.Var, in_col_x: gp.Var, in_col_y: gp.Var ,out_b: np.ndarray, out_r: np.ndarray, fwd: gp.Var, bwd: gp.Var):
    m.addConstr(NROW*in_col_u + gp.quicksum(out_b) <= NROW)
    m.addConstr(gp.quicksum(in_b) + gp.quicksum(out_b) - NBRANCH*in_col_x <= 2*NROW - NBRANCH)
    m.addConstr(gp.quicksum(in_b) + gp.quicksum(out_b) - 2*NROW*in_col_x >= 0)

    m.addConstr(NROW*in_col_u + gp.quicksum(out_r) <= NROW)
    m.addConstr(NROW*in_col_y == gp.quicksum(out_r))

    m.addConstr(gp.quicksum(out_b) - NROW * in_col_x - bwd == 0)
    m.addConstr(fwd == 0)
    m.update()

# generate XOR-MC rule, for 
def gen_XORMC_rule(m: gp.Model, in1_b: np.ndarray, in1_r: np.ndarray, in2_b: np.ndarray, in2_r: np.ndarray, col_u: gp.Var, col_x: gp.Var, col_y: gp.Var, col_z: gp.Var, col:np.ndarray, out_b: np.ndarray, out_r: np.ndarray, fwd: gp.Var, bwd: gp.Var):
    m.addConstr(NROW*col_u + gp.quicksum(out_b) <= NROW)
    m.addConstr(NROW*col_u + gp.quicksum(out_r) <= NROW)
    m.addConstr(gp.quicksum(out_b) - NROW* col_x == 0)

    m.addConstr(gp.quicksum(out_r) - gp.quicksum(col) - NBRANCH*col_y - col_u <= -1)
    m.addConstr(gp.quicksum(out_r) - gp.quicksum(col) - 2*NROW*col_y >= -1*NROW)

    m.addConstr(gp.quicksum(in1_r) + gp.quicksum(in2_r) <= 7 + col_z)
    m.addConstr(fwd ==  gp.quicksum(out_r) - NROW* col_z)
    m.addConstr(bwd == 0)
    m.update()

# generate matching rules, for easy calculation of dm
def gen_match_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_g: np.ndarray, out_b: np.ndarray, out_r: np.ndarray, out_g: np.ndarray, meet_signed, meet):
    m.addConstr(meet_signed == 
        gp.quicksum(in_b) + gp.quicksum(in_r) - gp.quicksum(in_g) +
        gp.quicksum(out_b) + gp.quicksum(out_r) - gp.quicksum(out_g) - NROW)
    m.addConstr(meet == gp.max_(meet_signed, 0))
    m.update()

# set objective function
def set_obj(m: gp.Model, ini_enc_b: np.ndarray, ini_enc_r: np.ndarray, ini_key_b: np.ndarray, ini_key_r: np.ndarray, cost_fwd: np.ndarray, cost_XOR: np.ndarray, cost_bwd: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray, meet: np.ndarray):
    df_b = m.addVar(lb=1, vtype=GRB.INTEGER, name="DF_b")
    df_r = m.addVar(lb=1, vtype=GRB.INTEGER, name="DF_r")
    dm = m.addVar(lb=1, vtype=GRB.INTEGER, name="Match")
    obj = m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

    m.addConstr(df_b == gp.quicksum(ini_enc_b.flatten()) + gp.quicksum(ini_key_b.flatten()) - gp.quicksum(cost_fwd.flatten()) - gp.quicksum(cost_XOR.flatten()) - gp.quicksum(key_cost_fwd.flatten()))
    m.addConstr(df_r == gp.quicksum(ini_enc_r.flatten()) + gp.quicksum(ini_key_r.flatten()) - gp.quicksum(cost_bwd.flatten()) - gp.quicksum(key_cost_bwd.flatten()))
    m.addConstr(dm == gp.quicksum(meet.flatten()))
    m.addConstr(obj - df_b <= 0)
    m.addConstr(obj - df_r <= 0)
    m.addConstr(obj - dm <= 0)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

def key_expansion(m:gp.Model, key_size:int, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, K_b: np.ndarray, K_r: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
    for r in range(0, total_r):
        # initial state of key expansion, strictly no unknown or consumed df
        if r == start_r:
            for i in ROW:
                for j in COL:
                    m.addConstr(K_b[r, i, j] + K_r[r, i, j] >= 1)
                    m.addConstr(K_ini_b[i, j] + K_r[r, i, j] == 1)
                    m.addConstr(K_ini_r[i, j] + K_b[r, i, j] == 1)
                    m.addConstr(key_cost_bwd[r,i,j] == 0)
                    m.addConstr(key_cost_fwd[r,i,j] == 0)
        
        # fwd direction
        if r > start_r:
            lr = r - 1
            for j in COL:
                if j == 0:  # special treatment for col 0 as in AES key schedule
                    for i in ROW:
                        rot_i = (i+1) % NROW
                        rot_j = NCOL - 1
                        gen_XOR_rule(m, in1_b=K_b[lr,i,j], in1_r=K_r[lr,i,j], in2_b=K_b[lr,rot_i,rot_j], in2_r=K_r[lr,rot_i,rot_j], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_fwd[r,i,j])
                        m.addConstr(key_cost_bwd[r,i,j] == 0)
                else:
                    for i in ROW:
                        gen_XOR_rule(m, in1_b=K_b[lr,i,j], in1_r=K_r[lr,i,j], in2_b=K_b[r,i,j-1], in2_r=K_r[r,i,j-1], out_b=K_b[r,i,j], out_r=K_r[r,i,j], cost_df= key_cost_fwd[r,i,j])
                        m.addConstr(key_cost_bwd[r,i,j] == 0)

        # bwd direction, reverse blue and red cell notation for XOR propagation        
        if r < start_r:  
            lr = r + 1
            for j in COL:
                if j == 0:  # special treatment for col 0 as in AES key schedule
                    for i in ROW:
                        rot_i = (i+1) % NROW
                        rot_j = NCOL - 1
                        gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[r,rot_i,rot_j], in2_r=K_b[r,rot_i,rot_j], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_bwd[r,i,j])
                        m.addConstr(key_cost_fwd[r,i,j] == 0)
                else:
                    for i in ROW:
                        gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[lr,i,j-1], in2_r=K_b[lr,i,j-1], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_bwd[r,i,j])
                        m.addConstr(key_cost_fwd[r,i,j] == 0)
    
    # store k[-1] at the end of the key state array, either bwd state or start state
    r = total_r
    if start_r == -1:
        for i in ROW:
            for j in COL:
                m.addConstr(K_b[r, i, j] + K_r[r, i, j] >= 1)
                m.addConstr(K_ini_b[i, j] + K_r[r, i, j] == 1)
                m.addConstr(K_ini_r[i, j] + K_b[r, i, j] == 1)
                m.addConstr(key_cost_bwd[r,i,j] == 0)
                m.addConstr(key_cost_fwd[r,i,j] == 0)
    else:
        lr = 0
        for j in COL:
            if j == 0:  # special treatment for col 0 as in AES key schedule
                for i in ROW:
                    rot_i = (i+1) % NROW
                    rot_j = NCOL - 1
                    gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[r,rot_i,rot_j], in2_r=K_b[r,rot_i,rot_j], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_bwd[r,i,j])
                    m.addConstr(key_cost_fwd[r,i,j] == 0)
            else:
                for i in ROW:
                    gen_XOR_rule(m, in1_b=K_r[lr,i,j], in1_r=K_b[lr,i,j], in2_b=K_r[lr,i,j-1], in2_r=K_b[lr,i,j-1], out_b=K_r[r,i,j], out_r=K_b[r,i,j], cost_df= key_cost_bwd[r,i,j])
                    m.addConstr(key_cost_fwd[r,i,j] == 0)

def writeSol(m: gp.Model, path):
    if m.SolCount > 0:
        if m.getParamInfo(GRB.Param.PoolSearchMode)[2] > 0:
            gv = m.getVars()
            names = m.getAttr('VarName', gv)
            for i in range(m.SolCount):
                m.params.SolutionNumber = i
                xn = m.getAttr('Xn', gv)
                lines = ["{} {}".format(v1, v2) for v1, v2 in zip(names, xn)]
                with open('./runlog/{}_{}.sol'.format(m.modelName, i), 'w') as f:
                    f.write("# Solution for model {}\n".format(m.modelName))
                    f.write("# Objective value = {}\n".format(m.PoolObjVal))
                    f.write("\n".join(lines))
        else:
            m.write(path + m.modelName + '.sol')
        return 1
    else:
        return 0

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
    def headliner(r:int, f: TextIOWrapper):
        header = "r%d  " %r 
        sign = ''
        
        if r == match_round:
            header+= '-><-'
        elif r in fwd:
            header+= '--->'
        elif r in bwd:
            header+= '<---'

        if r == enc_start:
            sign+='ENC '
        if r == match_round:
            sign = "{:<30}".format(sign)
            sign+='MAT '
        if r == key_start:
            sign = "{:<36}".format(sign)
            sign += 'KEY '
        
        f.write(header + '\n' + sign + '\n')

    if not os.path.exists(path= path + m.modelName +'.sol'):
        return

    solFile = open(path + m.modelName +'.sol', 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
    
    match = re.match(r'AES(\d+)RK_(\d+)r_ENC_r(\d+)_KEY_r(\d+)Meet_r(\d+)', m.modelName)
    key_size, total_round, enc_start, key_start, match_round = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))

    if enc_start < match_round:
        fwd = list(range(enc_start, match_round))
        bwd = list(range(match_round + 1, total_round)) + list(range(0, enc_start))
    else:
        bwd = list(range(match_round + 1, enc_start))
        fwd = list(range(enc_start, total_round)) + list(range(0, match_round))
        

    SB_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    SB_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    AK_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    AK_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    KEY_b= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    KEY_r= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    cost_xor = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    cost_fwd = np.ndarray(shape=(total_round, NCOL), dtype=int)
    cost_bwd = np.ndarray(shape=(total_round, NCOL), dtype=int)

    ini_enc_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_g = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_g = np.ndarray(shape=(NROW, NCOL), dtype=int)

    meet =  np.ndarray(shape=(NCOL), dtype=int)
    meet_s =  np.ndarray(shape=(NCOL), dtype=int)

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                SB_b[r,i,j]=Sol["S_b[%d,%d,%d]" %(r,i,j)]
                SB_r[r,i,j]=Sol["S_r[%d,%d,%d]" %(r,i,j)]
                AK_b[r,i,j]=Sol["A_b[%d,%d,%d]" %(r,i,j)]
                AK_r[r,i,j]=Sol["A_r[%d,%d,%d]" %(r,i,j)]

    for ri in range(total_round):
        for i in ROW:
            for j in COL:
                MC_b[ri, i, j] = SB_b[ri, i, (j + i)%NCOL]
                MC_r[ri, i, j] = SB_r[ri, i, (j + i)%NCOL]

    for r in range(total_round+1):
        for i in ROW:
            for j in COL:
                KEY_b[r,i,j]=Sol["K_b[%d,%d,%d]" %(r,i,j)]
                KEY_r[r,i,j]=Sol["K_r[%d,%d,%d]" %(r,i,j)]
                cost_xor[r,i,j] = Sol["Cost_XOR[%d,%d,%d]" %(r,i,j)]
    
    for r in range(total_round):
        for j in COL:
            cost_fwd[r,j] = Sol["Cost_fwd[%d,%d]" %(r,j)]
            cost_bwd[r,j] = Sol["Cost_bwd[%d,%d]" %(r,j)]

    for i in ROW:
        for j in COL:
            ini_enc_b[i,j] = Sol["S_ini_b[%d,%d]" %(i,j)]
            ini_enc_r[i,j] = Sol["S_ini_r[%d,%d]" %(i,j)]
            ini_enc_g[i,j] = Sol["S_ini_g[%d,%d]" %(i,j)]
            ini_key_b[i,j] = Sol["K_ini_b[%d,%d]" %(i,j)]
            ini_key_r[i,j] = Sol["K_ini_r[%d,%d]" %(i,j)]
            ini_key_g[i,j] = Sol["K_ini_g[%d,%d]" %(i,j)]
    
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
    f.write(m.modelName + '\n')
    f.write('ENC FWD: ' + str(ini_df_enc_b) + '\n' + 'ENC BWD: ' + str(ini_df_enc_r) + '\n')
    f.write('KEY FWD: ' + str(ini_df_key_b) + '\n' + 'KEY BWD: ' + str(ini_df_key_r) + '\n')
    f.write('Obj= min{DF_b=%d, DF_r=%d, Match=%d} = %d' %(DF_b, DF_r, Match, Obj) + '\n')
    f.write('\n')

    for r in range(total_round):
        headliner(r, f)
        if r == match_round:
            f.write('SB#%d'%r +TAB*2+'MC#%d' %r +TAB*2+'MC^K' +TAB*2+ 'SB#%d' % (r+1) + TAB*2 +'K#%d ' %r +'\n')
            for i in ROW:
                SB = ''
                MC = ''
                AK = ''
                KEY = ''
                NSB=''
                for j in COL:
                    SB+=color(SB_b[r,i,j], SB_r[r,i,j])
                    MC+=color(MC_b[r,i,j], MC_r[r,i,j])
                    KEY+=color(KEY_b[r,i,j], KEY_r[r,i,j])
                    NSB+=color(SB_b[r+1,i,j], SB_r[r+1,i,j])
                    AK+=color(AK_b[r,i,j], AK_r[r,i,j])
                
                f.write(SB+TAB*2+MC+TAB*2+AK+TAB*2+NSB+TAB*2+KEY+'\n') 
            f.write('Meet_signed' + str(meet_s[:]) + '\n')
            f.write('Meet' + str(meet[:]) + '\n')
        else:
            f.write('SB#%d'%r +TAB*2+'MC#%d' %r +TAB*2+'AK#%d' %r +TAB*2+'K#%d ' %r +'\n')
        
            for i in ROW:
                SB = ''
                MC = ''
                AK = ''
                KEY = ''
                
                for j in COL:
                    SB+=color(SB_b[r,i,j], SB_r[r,i,j])
                    MC+=color(MC_b[r,i,j], MC_r[r,i,j])
                    KEY+=color(KEY_b[r,i,j], KEY_r[r,i,j])
                    if r == total_round -1 or r in bwd:
                        AK+=' '
                    else:
                        AK+=color(AK_b[r,i,j], AK_r[r,i,j])
                
                f.write(SB+TAB*2+MC+TAB*2+AK+TAB*2+KEY+'\n')   
        
        f.write('CostDF fwd: '+ str(cost_fwd[r,:]) + '\n')
        f.write('CostDF XOR: ' + '\n' + str(cost_xor[r,:,:]) + '\n')
        f.write('CostDF bwd: '+ str(cost_bwd[r,:]) + '\n')
        f.write('\n')

    # process whiten key
    r = -1
    f.write("r%d  " %r + '\n')
    f.write(6*TAB +'AT  '+ TAB*2 + 'K#-1' + '\n')
    for i in ROW:
        KEY = ''
        AT = ''
        for j in COL:
            KEY+=color(KEY_b[r,i,j], KEY_r[r,i,j])
            AT +=color(AK_b[r+total_round,i,j], AK_r[r+total_round,i,j])
        f.write(6*TAB + AT+ TAB*2 + KEY + '\n')
    
    f.write('CostDF fwd: '+ str(cost_fwd[r+total_round,:]) + '\n')
    f.write('CostDF XOR: ' + '\n' + str(cost_xor[r+total_round,:,:]) + '\n')
    f.write('CostDF bwd: '+ str(cost_bwd[r+total_round,:]) + '\n')
    f.close()

    return 'Obj= min{DF_b=%d, DF_r=%d, Match=%d} = %d' %(DF_b, DF_r, Match, Obj)
####################################################################################################################

# interable
def solve(key_size:int, total_round:int, start_round:int, key_start_round:int, match_round:int):
    # define optimization model
    m = gp.Model('AES%dRK_%dr_ENC_r%d_KEY_r%dMeet_r%d' % (key_size, total_round, start_round, key_start_round, match_round))
    
    # assign forward and backward rounds, excluding match round and last round
    if start_round < match_round:
        fwd = list(range(start_round, match_round))
        bwd = list(range(match_round+1, total_round-1)) + list(range(0, start_round))
    else:
        bwd = list(range(match_round+1, start_round))
        fwd = list(range(start_round, total_round-1)) + list(range(0, match_round))

    # registration of variables
    [   S_b, S_r, S_g, S_w, M_b, M_r, M_g, M_w, A_b, A_r, A_g, A_w, K_b, K_r, K_g, K_w,
    S_col_u, S_col_x, S_col_y, M_col_u, M_col_x, M_col_y, K_col_u, K_col_x, K_col_y, XorMC, XorMC_u, XorMC_x, XorMC_y, XorMC_z,
    E_ini_b, E_ini_r, E_ini_g, K_ini_b, K_ini_r, K_ini_g,
    cost_fwd, cost_XOR, cost_bwd, key_cost_fwd, key_cost_bwd, meet_signed, meet] = def_var(m, total_round, fwd, bwd)

    # add constriants according to the key expansion algorithm
    key_expansion(m, key_size, total_round, key_start_round, K_ini_b, K_ini_r, K_b, K_r, key_cost_fwd, key_cost_bwd)

    # initialize the enc states, avoid unknown to maximize performance
    for i in ROW:
        for j in COL:
            m.addConstr(S_b[start_round, i, j] + S_r[start_round, i, j] >= 1)
            m.addConstr(E_ini_b[i, j] + S_r[start_round, i, j] == 1)
            m.addConstr(E_ini_r[i, j] + S_b[start_round, i, j] == 1)

    # add constriants according to the encryption algorithm
    for r in range(total_round):
        nr = r + 1   # alias for next round
        # match round
        if r == match_round:
            print('mat', r)
            # use tempK to store the key state after the inverse MC operation, include cost of df
            tempK_b = np.asarray(m.addVars(NROW, NCOL, vtype= GRB.BINARY, name='tempK_b').values()).reshape((NROW, NCOL))
            tempK_r = np.asarray(m.addVars(NROW, NCOL, vtype= GRB.BINARY, name='tempK_r').values()).reshape((NROW, NCOL))
            for j in COL:    
                gen_MC_rule(m, K_b[r,:,j], K_r[r,:,j], K_col_u[r,j], K_col_x[r,j], K_col_y[r,j], tempK_b[:,j], tempK_r[:,j], cost_fwd[r,j], cost_bwd[r,j])      
            # use AK to store MC state after XOR with tempK (different from other rounds, take carefully note)
            for i in ROW:
                for j in COL:
                    gen_XOR_rule(m, M_b[r,i,j], M_r[r,i,j], tempK_b[i,j], tempK_r[i,j], A_b[r,i,j], A_r[r,i,j], cost_XOR[r,i,j])
            # meet-in-the-middle for AK == MC[r] XOR MC^-1(KEY[r]), and SB[nr]
            for j in COL:
                gen_match_rule(m, A_b[r,:,j], A_r[r,:,j], A_g[r,:,j], S_b[nr,:,j], S_r[nr,:,j], S_g[nr,:,j], meet_signed[j], meet[j])
                #m.addConstr(cost_fwd[r,j] == 0)
                #m.addConstr(cost_bwd[r,j] == 0)
        # last round
        elif r == total_round - 1:
            print('lastr', r)
            for j in COL:
                m.addConstr(cost_fwd[r, j] == 0)
                m.addConstr(cost_bwd[r, j] == 0)
                # skip the MC for last round, use AK to store id(MC[lr]) XOR KEY[lr]
                for i in ROW:
                    gen_XOR_rule(m, M_b[r,i,j], M_r[r,i,j], K_b[r,i,j], K_r[r,i,j], A_b[r,i,j], A_r[r,i,j], cost_XOR[r,i,j])
                # add whitening key: AK[lr] XOR KEY[-1] (stored as KEY[tr]) should equal to SB[0]
                for i in ROW:
                    gen_XOR_rule(m, A_b[r,i,j], A_r[r,i,j], K_b[nr,i,j], K_r[nr,i,j], S_b[0,i,j], S_r[0,i,j], cost_XOR[nr,i,j])
        # forward direction
        elif r in fwd:
            print('fwd', r)
            for j in COL:
                gen_MC_rule(m, M_b[r,:,j], M_r[r,:,j], M_col_u[r,j], M_col_x[r,j], M_col_y[r,j], A_b[r,:,j], A_r[r,:,j], cost_fwd[r,j], cost_bwd[r,j])
                for i in ROW:
                    gen_XOR_rule(m, A_b[r,i,j], A_r[r,i,j], K_b[r,i,j], K_r[r,i,j], S_b[nr,i,j], S_r[nr,i,j], cost_XOR[r,i,j])
        # backward direction
        elif r in bwd:
            print('bwd', r)
            for j in COL:
                gen_XORMC_rule(m, S_b[nr,:,j], S_r[nr,:,j], K_b[r,:,j], K_r[r,:,j], XorMC_u[r,j], XorMC_x[r,j], XorMC_y[r,j], XorMC_z[r,j], XorMC[r,:,j], M_b[r,:,j], M_r[r,:,j], cost_fwd[r,j], cost_bwd[r,j])
                for i in ROW:
                    m.addConstr(cost_XOR[r,i,j] == 0)
        else:
            raise Exception("Irregular Behavior!")
    
    # set objective function
    set_obj(m, E_ini_b, E_ini_r, K_ini_b, K_ini_r, cost_fwd, cost_XOR, cost_bwd, key_cost_fwd, key_cost_bwd, meet)
    m.optimize()
    
    dir = './RK_output/'
    if not os.path.exists(path= dir):
        os.makedirs(dir)
    
    if writeSol(m, path=dir):
        solution = displaySol(m, path=dir)
        return m.modelName + TAB + str(solution)
    else:
        return m.modelName + TAB + 'Infeasible'
solve(key_size=128, total_round=8, start_round=4, key_start_round=4, match_round=1)
    




