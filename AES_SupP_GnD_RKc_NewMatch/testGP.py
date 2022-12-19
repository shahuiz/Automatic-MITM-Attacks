import numpy as np
import gurobipy as gp
from gurobipy import GRB

# A and B are constriants for XOR operations, derived by SAGEMATH
# all valid i/o vectors must satisfy: Ax + B >= 0
# i/o vectors are formulated as: (in1_b, in1_r, in2_b, in2_r, out_b, out_r, cost_df)


# example usage of the XOR classifier
m = gp.Model('x')
match_case_1 = m.addVar(vtype = GRB.BINARY, name='match_case_1')
x1 = m.addVar(vtype = GRB.BINARY)
x2 = m.addVar(vtype = GRB.BINARY)
x4 = m.addVar(vtype = GRB.BINARY)
x7 = m.addVar(vtype = GRB.BINARY)
obj = m.addVar(vtype = GRB.BINARY)
m.addConstr(match_case_1 == gp.max_(x1*x2, x4))
m.addGenConstrIndicator(x7, True, x1 + 2*x2 + x4, GRB.EQUAL, 1.0)

m.update()
m.write('./x.lp')
m.optimize()
#print(m.getAttr("match_case_1"))
m.write('./x.sol')
