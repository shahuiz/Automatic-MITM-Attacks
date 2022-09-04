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
total_r = 7 # total round
start_round = 4   # start round, start in {0,1,2,...,total_r-1}
match_round = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid

m = gp.Model('model_%dx%d_%dR_Start_r%d_Meet_r%d' % (NROW, NCOL, total_r, start_round, match_round))

S_b = np.asarray(m.addVars(total_r, NROW, NCOL, vtype= GRB.BINARY, name='S_b').values()).reshape((total_r, NROW, NCOL))

m.update()

m.addLConstr(gp.quicksum(S_b[0,0,:]) - gp.quicksum(S_b[0,1,:]), GRB.EQUAL, 3)
m.update()
print(m.getConstrByName("R0"))

print(S_b[0,0,0])