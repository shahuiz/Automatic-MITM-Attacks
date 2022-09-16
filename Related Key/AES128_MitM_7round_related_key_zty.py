from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import result_vis as vis

# AES parameters
NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)

# variable declaration
total_round = 7 # total round
start_round = 4   # start round, start in {0,1,2,...,total_r-1}
match_round = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid
key_start_round = 4 

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

m = gp.Model('model_%dx%d_%dR_Start_r%d_Meet_r%d_RelatedKey' % (NROW, NCOL, total_round, start_round, match_round))

def def_var(total_r: int, m:gp.Model):
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

    for r in bwd:
        nr = (r+1) % total_round
        for j in COL:
            m.addConstr(XorMC_u[r,j] == gp.or_(S_col_u[nr,j], K_col_u[r,j]))
            m.addConstr(XorMC_x[r,j] == gp.and_(S_col_x[nr,j], K_col_x[r,j]))
            m.addConstr(XorMC_y[r,j] == gp.and_(S_col_y[nr,j], K_col_y[r,j]))
            m.addConstr(XorMC_z[r,j] == gp.min_(S_r[nr,:,j].tolist() + K_r[r,:,j].tolist()))
            for i in ROW:
                m.addConstr(XorMC[r,i,j] == gp.or_(S_b[nr,i,j], K_r[r,i,j]))

    m.update()
    return [
        S_b, S_r, S_g, S_w, 
        M_b, M_r, M_g, M_w, 
        A_b, A_r, A_g, A_w,
        K_b, K_r, K_g, K_w,
        S_col_u, S_col_x, S_col_y, 
        M_col_u, M_col_x, M_col_y, 
        K_col_u, K_col_x, K_col_y,
        XorMC, XorMC_u, XorMC_x, XorMC_y, XorMC_z,
        S_ini_b, S_ini_r, S_ini_g,
        K_ini_b, K_ini_r, K_ini_g,
        cost_fwd, cost_XOR, cost_bwd, 
        key_cost_fwd, key_cost_bwd,
        meet_signed, meet]

# generate XOR rule for forward computations, if backward, switch the input of blue and red
def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_df: gp.Var):
    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df]
    m.addMConstr(XOR_A, list(enum), '>=', -XOR_B)

# generate MC rule 
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

# generate XOR-MC rule
def gen_XORMC_rule(m: gp.Model, in1_b: np.ndarray, in1_r: np.ndarray, in2_b: np.ndarray, in2_r: np.ndarray, col_u: gp.Var, col_x: gp.Var, col_y: gp.Var, col_z: gp.Var, col:np.ndarray, out_b: np.ndarray, out_r: np.ndarray, bwd: gp.Var):
    m.addConstr(NROW*col_u + gp.quicksum(out_b) <= NROW)
    m.addConstr(NROW*col_u + gp.quicksum(out_r) <= NROW)
    m.addConstr(gp.quicksum(out_b) - NROW* col_x == 0)

    m.addConstr(gp.quicksum(out_r) - gp.quicksum(col) - NBRANCH*col_y - col_u <= -1)
    m.addConstr(gp.quicksum(out_r) - gp.quicksum(col) - 2*NROW*col_y >= -1*NROW)

    m.addConstr(gp.quicksum(in1_r) + gp.quicksum(in2_r) <= 7 + col_z)
    m.addConstr(bwd == -4* col_z + gp.quicksum(out_r))
    m.update()

# generate matching rules, for easy calculation of dm
def gen_match_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_g: np.ndarray, out_b: np.ndarray, out_r: np.ndarray, out_g: np.ndarray, meet_signed, meet):
    m.addConstr(meet_signed == 
        gp.quicksum(in_b) + gp.quicksum(in_r) - gp.quicksum(in_g) +
        gp.quicksum(out_b) + gp.quicksum(out_r) - gp.quicksum(out_g) - NROW)
    m.addConstr(meet == gp.max_(meet_signed, 0))
    m.update()

# set objective function
def set_obj(m: gp.Model, start_b: np.ndarray, start_r: np.ndarray, cost_fwd: np.ndarray, cost_bwd: np.ndarray, meet: np.ndarray):
    df_b = m.addVar(lb=1, vtype=GRB.INTEGER, name="Final_b")
    df_r = m.addVar(lb=1, vtype=GRB.INTEGER, name="Final_r")
    dm = m.addVar(lb=1, vtype=GRB.INTEGER, name="Match")
    obj = m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

    m.addConstr(df_b == gp.quicksum(start_b.flatten()) - gp.quicksum(cost_fwd.flatten()) - gp.quicksum(cost_XOR.flatten()))
    m.addConstr(df_r == gp.quicksum(start_r.flatten()) - gp.quicksum(cost_bwd.flatten()))
    m.addConstr(dm == gp.quicksum(meet.flatten()))
    m.addConstr(obj - df_b <= 0)
    m.addConstr(obj - df_r <= 0)
    m.addConstr(obj - dm <= 0)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

def key_expansion(m:gp.Model, total_r: int, start_r: int, K_ini_b: np.ndarray, K_ini_r: np.ndarray, K_b: np.ndarray, K_r: np.ndarray, key_cost_fwd: np.ndarray, key_cost_bwd: np.ndarray):
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

####################################################################################################################
# main
fwd = []    # forward rounds
bwd = []    # backward rounds

if start_round < match_round:
    fwd = list(range(start_round, match_round))
    bwd = list(range(match_round + 1, total_round)) + list(range(0, start_round))
else:
    bwd = list(range(match_round + 1, start_round))
    fwd = list(range(start_round, total_round)) + list(range(0, match_round))

print(fwd)
print(bwd)

[   S_b, S_r, S_g, S_w, 
    M_b, M_r, M_g, M_w, 
    A_b, A_r, A_g, A_w,
    K_b, K_r, K_g, K_w,
    S_col_u, S_col_x, S_col_y, 
    M_col_u, M_col_x, M_col_y, 
    K_col_u, K_col_x, K_col_y,
    XorMC, XorMC_u, XorMC_x, XorMC_y, XorMC_z,
    S_ini_b, S_ini_r, S_ini_g,
    K_ini_b, K_ini_r, K_ini_g,
    cost_fwd, cost_XOR, cost_bwd, 
    key_cost_fwd, key_cost_bwd,
    meet_signed, meet] = def_var(total_round, m)

key_expansion(m, total_round, start_round, K_ini_b, K_ini_r, K_b, K_r, key_cost_fwd, key_cost_bwd)

# main function
for r in range(total_round):
    nr = (r+1) % total_round
    if r == start_round:
        for i in ROW:
            for j in COL:
                m.addConstr(S_b[r, i, j] + S_r[r, i, j] >= 1)
                m.addConstr(S_ini_b[i, j] + S_r[r, i, j] == 1)
                m.addConstr(S_ini_r[i, j] + S_b[r, i, j] == 1)
    if r == match_round:
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
            m.addConstr(cost_fwd[r,j] == 0)
            m.addConstr(cost_bwd[r,j] == 0)
    else:
        if r == total_round - 1:
            tr = r + 1
            for j in COL:
                m.addConstr(cost_fwd[r, j] == 0)
                m.addConstr(cost_bwd[r, j] == 0)
                # jump the MC for last round, use AK to store id(MC[lr]) XOR KEY[lr]
                for i in ROW:
                    gen_XOR_rule(m, M_b[r,i,j], M_r[r,i,j], K_b[r,i,j], K_r[r,i,j], A_b[r,i,j], A_r[r,i,j], cost_XOR[r,i,j])
                # add whitening key: AK[lr] XOR KEY[-1] (stored as KEY[tr]) should equal to SB[0]
                for i in ROW:
                    gen_XOR_rule(m, A_b[r,i,j], A_r[r,i,j], K_b[tr,i,j], K_r[tr,i,j], S_b[0,i,j], S_r[0,i,j], cost_XOR[tr,i,j])

        elif r in fwd:
            print('fwd', r)
            for j in COL:
                gen_MC_rule(m, M_b[r,:,j], M_r[r,:,j], M_col_u[r,j], M_col_x[r,j], M_col_y[r,j], A_b[r,:,j], A_r[r,:,j], cost_fwd[r,j], cost_bwd[r,j])
                for i in ROW:
                    gen_XOR_rule(m, A_b[r,i,j], A_r[r,i,j], K_b[r,i,j], K_r[r,i,j], S_b[nr,i,j], S_r[nr,i,j], cost_XOR[r,i,j])
        elif r in bwd:
            print('bwd', r)
            for j in COL:
                gen_XORMC_rule(m, S_b[nr,:,j], S_r[nr,:,j], K_b[r,:,j], K_r[r,:,j], XorMC_u[r,j], XorMC_x[r,j], XorMC_y[r,j], XorMC_z[r,j], XorMC[r,:,j], M_b[r,:,j], M_r[r,:,j], cost_bwd[r,j])
                for i in ROW:
                    m.addConstr(cost_XOR[r,i,j] == 0)
                
set_obj(m, S_ini_b, S_ini_r, cost_fwd, cost_bwd, meet)
m.optimize()
#writeSol()
print(m)

fnp = './runlog/' + m.modelName + '.sol'
vis.writeSol(m)