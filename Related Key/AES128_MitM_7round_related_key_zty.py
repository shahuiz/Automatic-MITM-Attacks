from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB
import numpy as np

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

    # define vars storing the key state at each round with encoding scheme
    K_b = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='K_b').values()).reshape((total_r, NROW, NCOL))
    K_r = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='K_r').values()).reshape((total_r, NROW, NCOL))
    K_g = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='K_g').values()).reshape((total_r, NROW, NCOL))
    K_w = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='K_w').values()).reshape((total_r, NROW, NCOL))

    # define variables for columnwise encoding
    S_col_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_u').values()).reshape((total_r, NCOL))
    S_col_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_x').values()).reshape((total_r, NCOL))
    S_col_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_y').values()).reshape((total_r, NCOL))

    M_col_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_u').values()).reshape((total_r, NCOL))
    M_col_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_x').values()).reshape((total_r, NCOL))
    M_col_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_y').values()).reshape((total_r, NCOL))

    # define vars to track the start state of Encryption states (S) and Key states (K)
    S_ini_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='S_ini_b').values()).reshape((NROW, NCOL))
    S_ini_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='S_ini_r').values()).reshape((NROW, NCOL))
    S_ini_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='S_ini_g').values()).reshape((NROW, NCOL))
    
    K_ini_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_b').values()).reshape((NROW, NCOL))
    K_ini_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_r').values()).reshape((NROW, NCOL))
    K_ini_g = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='K_ini_g').values()).reshape((NROW, NCOL))
    
    # define auxiliary vars tracking cost of df at MC operations
    cost_fwd = np.asarray(m.addVars(total_r, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Cost_fwd').values()).reshape((total_r, NCOL))
    cost_bwd = np.asarray(m.addVars(total_r, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Cost_bwd').values()).reshape((total_r, NCOL))

    key_cost_fwd = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_fwd').values()).reshape((total_r, NROW, NCOL))
    key_cost_bwd = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='Key_cost_bwd').values()).reshape((total_r, NROW, NCOL))
    
    # define auxiliary vars for computations on degree of matching
    meet_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype=GRB.INTEGER, name='Meet_signed').values())
    meet = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Meet').values())
    
    m.update()
    return [
        S_b, S_r, S_g, S_w, 
        M_b, M_r, M_g, M_w, 
        K_b, K_r, K_g, K_w,
        S_col_u, S_col_x, S_col_y, 
        M_col_u, M_col_x, M_col_y, 
        S_ini_b, S_ini_r, S_ini_g,
        K_ini_b, K_ini_r, K_ini_g,
        cost_fwd, cost_bwd, 
        key_cost_fwd, key_cost_bwd,
        meet_signed, meet]

# define encode rules
def gen_encode_rule(m: gp.Model, total_r: int, S_b: np.ndarray, S_r: np.ndarray, S_g: np.ndarray, S_w: np.ndarray, M_b: np.ndarray, M_r: np.ndarray, M_g: np.ndarray, M_w: np.ndarray, S_col_u: np.ndarray, S_col_x: np.ndarray, S_col_y: np.ndarray, M_col_u: np.ndarray, M_col_x: np.ndarray, M_col_y: np.ndarray):
    for r in range(total_r):
        for i in ROW:
            for j in COL:
                m.addConstr(S_g[r,i,j] == gp.and_(S_b[r,i,j], S_r[r,i,j]))
                m.addConstr(S_w[r,i,j] + S_b[r,i,j] + S_r[r,i,j] - S_g[r,i,j] == 1)

    for r in range(total_r):
        for j in COL:
            m.addConstr(S_col_x[r,j] == gp.min_(S_b[r,:,j].tolist()))
            m.addConstr(S_col_y[r,j] == gp.min_(S_r[r,:,j].tolist()))
            m.addConstr(S_col_u[r,j] == gp.max_(S_w[r,:,j].tolist()))

            m.addConstr(M_col_x[r,j] == gp.min_(M_b[r,:,j].tolist()))
            m.addConstr(M_col_y[r,j] == gp.min_(M_r[r,:,j].tolist()))
            m.addConstr(M_col_u[r,j] == gp.max_(M_w[r,:,j].tolist()))
    m.update()

# define XOR rule for forward computations, if backward, switch the input of blue and red
def gen_XOR_rule(m: gp.Model, in1_b: gp.Var, in1_r: gp.Var, in2_b: gp.Var, in2_r: gp.Var, out_b: gp.Var, out_r: gp.Var, cost_df: gp.Var):
    enum = [in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df]
    m.addMConstr(XOR_A, list(enum), '>=', -XOR_B)

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

def gen_match_rule(m: gp.Model, in_b: np.ndarray, in_r: np.ndarray, in_g: np.ndarray, out_b: np.ndarray, out_r: np.ndarray, out_g: np.ndarray, meet_signed, meet):
    m.addConstr(meet_signed == 
        gp.quicksum(in_b) + gp.quicksum(in_r) - gp.quicksum(in_g) +
        gp.quicksum(out_b) + gp.quicksum(out_r) - gp.quicksum(out_g) - NROW)
    m.addConstr(meet == gp.max_(meet_signed, 0))
    m.update()

def set_obj(m: gp.Model, start_b: np.ndarray, start_r: np.ndarray, cost_fwd: np.ndarray, cost_bwd: np.ndarray, meet: np.ndarray):
    df_b = m.addVar(lb=1, vtype=GRB.INTEGER, name="Final_b")
    df_r = m.addVar(lb=1, vtype=GRB.INTEGER, name="Final_r")
    dm = m.addVar(lb=1, vtype=GRB.INTEGER, name="Match")
    obj = m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

    m.addConstr(df_b == gp.quicksum(start_b.flatten()) - gp.quicksum(cost_fwd.flatten()))
    m.addConstr(df_r == gp.quicksum(start_r.flatten()) - gp.quicksum(cost_bwd.flatten()))
    m.addConstr(dm == gp.quicksum(meet.flatten()))
    m.addConstr(obj - df_b <= 0)
    m.addConstr(obj - df_r <= 0)
    m.addConstr(obj - dm <= 0)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

def key_expansion(m:gp.Model, K_ini_b: np.ndarray, K_ini_r: np.ndarray, K_b: np.ndarray, K_r: np.ndarray, total_r: int, start_r: int):
    for r in range(total_r):
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
    K_b, K_r, K_g, K_w,
    S_col_u, S_col_x, S_col_y, 
    M_col_u, M_col_x, M_col_y, 
    S_ini_b, S_ini_r, S_ini_g,
    K_ini_b, K_ini_r, K_ini_g,
    cost_fwd, cost_bwd, 
    key_cost_fwd, key_cost_bwd,
    meet_signed, meet] = def_var(total_round, m)

gen_encode_rule(m, total_round, S_b, S_r, S_g, S_w, M_b, M_r, M_g, M_w, S_col_u, S_col_x, S_col_y, M_col_u, M_col_x, M_col_y)

for r in range(total_round):
    nr = (r+1) % total_round
    if r == start_round:
        for i in ROW:
            for j in COL:
                m.addConstr(S_b[r, i, j] + S_r[r, i, j] >= 1)
                m.addConstr(S_ini_b[i, j] + S_r[r, i, j] == 1)
                m.addConstr(S_ini_r[i, j] + S_b[r, i, j] == 1)
    if r == match_round:
        for j in COL:
            gen_match_rule(m, M_b[r,:,j], M_r[r,:,j], M_g[r,:,j], S_b[nr,:,j], S_r[nr,:,j], S_g[nr,:,j], meet_signed[j], meet[j])
            m.addConstr(cost_fwd[r,j] == 0)
            m.addConstr(cost_bwd[r,j] == 0)
    else:
        if r == total_round - 1:
            for j in COL:
                m.addConstr(cost_fwd[r, j] == 0)
                m.addConstr(cost_bwd[r, j] == 0)
                # jump the MC for last round
                for i in ROW:
                    m.addConstr(M_b[r, i, j] - S_b[nr, i, j] == 0)
                    m.addConstr(M_r[r, i, j] - S_r[nr, i, j] == 0)
        elif r in fwd:
            print('fwd', r)
            for j in COL:
                gen_MC_rule(m, M_b[r,:,j], M_r[r,:,j], M_col_u[r,j], M_col_x[r,j], M_col_y[r,j], S_b[nr,:,j], S_r[nr,:,j], cost_fwd[r,j], cost_bwd[r,j])
        elif r in bwd:
            print('bwd', r)
            for j in COL:
                gen_MC_rule(m, S_b[nr,:,j], S_r[nr,:,j], S_col_u[nr,j], S_col_x[nr,j], S_col_y[nr,j], M_b[r,:,j], M_r[r,:,j], cost_fwd[r,j], cost_bwd[r,j])

set_obj(m, S_ini_b, S_ini_r, cost_fwd, cost_bwd, meet)
m.optimize()
#writeSol()
print(m)

fnp = './runlog/' + m.modelName + '.sol'
#drawSol(total_r=7, ini_r= 4, mat_r=1, F_r= fwd, B_r=bwd, outfile= fnp)