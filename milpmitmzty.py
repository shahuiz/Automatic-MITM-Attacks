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
start_r = 4   # start round, start in {0,1,2,...,total_r-1}
match_r = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid
fwd = []    # forward rounds
bwd = []    # backward rounds

if start_r < match_r:
    fwd = list(range(start_r, match_r))
    bwd = list(range(match_r + 1, total_r)) + list(range(0, start_r))
else:
    bwd = list(range(match_r + 1, start_r))
    fwd = list(range(start_r, total_r)) + list(range(0, match_r))

# initialize model
m = gp.Model('model_%dx%d_%dR_Start_r%d_Meet_r%d' % (NROW, NCOL, total_r, start_r, match_r))

# create Encode class to encode coloring scheme
class CellEncode(object):
    def __init__(self, x: gp.Var, y:gp.Var, z:gp.Var, a:gp.Var):
        self.b = x
        self.r = y
        self.g = z
        self.w = a

# define function to retrieve column wise information
def get_col(A: np.ndarray, cell: str):
    result = np.ndarray(shape=(NROW), dtype= gp.Var)
    for i in range(NROW):
        if cell == 'b':
            result[i] = A[i].b
        if cell == 'r':
            result[i] = A[i].r
        if cell == 'w':
            result[i] = A[i].w
        if cell == 'g':
            result[i] = A[i].g
    return result

# define variables to represent state pattern, with encoding scheme
S = np.ndarray(shape=(total_r, NROW, NCOL), dtype= CellEncode)  # store the SB state at each round
M = np.ndarray(shape=(total_r, NROW, NCOL), dtype= CellEncode)  # store the MC state at each round
for r in range(total_r):
    for i in ROW:
        for j in COL:
            S[r, i, j] = CellEncode(
                m.addVar(vtype=GRB.BINARY, name= "R%d[%d,%d]_b" %(r,i,j)),
                m.addVar(vtype=GRB.BINARY, name= "R%d[%d,%d]_r" %(r,i,j)),
                m.addVar(vtype=GRB.BINARY, name= "R%d[%d,%d]_g" %(r,i,j)),
                m.addVar(vtype=GRB.BINARY, name= "R%d[%d,%d]_w" %(r,i,j))
            )
for r in range(total_r):
    for i in ROW:
        for j in COL:
            M[r, i, j] = S[r, i, (j+i)%NCOL]    # match SB and MC through SR  

# generate encoding rule constraints
for r in range(total_r):
    for i in ROW:
        for j in COL:
            m.addConstr(S[r,i,j].g == gp.and_(S[r,i,j].b, S[r,i,j].r))  # grey cell
            m.addConstr(S[r,i,j].w - S[r,i,j].g + S[r,i,j].b + S[r,i,j].r == 1) # white cell 

# define the degree of freedom at the starting position
start_df_b = np.ndarray(shape = (NROW, NCOL), dtype = gp.Var)
start_df_r = np.ndarray(shape = (NROW, NCOL), dtype = gp.Var)

for i in ROW:
    for j in COL:
        start_df_b[i,j] = m.addVar(vtype=GRB.BINARY, name= "Start[%d,%d]_b" %(i,j))
        start_df_r[i,j] = m.addVar(vtype=GRB.BINARY, name= "Start[%d,%d]_r" %(i,j))

# define consumed degree of freedom along corresponding direction
consumed_fwd = np.ndarray(shape = (total_r, NCOL), dtype = gp.Var)
consumed_bwd = np.ndarray(shape = (total_r, NCOL), dtype = gp.Var)

for r in range(total_r):
    for j in COL:
        consumed_fwd[r, j] = m.addVar(lb=0, ub=NROW, vtype=GRB.INTEGER, name= "R%dC%d_fwd" %(r,j))
        consumed_bwd[r, j] = m.addVar(lb=0, ub=NROW, vtype=GRB.INTEGER, name= "R%dC%d_bwd" %(r,j))

# define intermediate values for computations on degree of matching
match_col_df_u = np.ndarray(shape=(NCOL), dtype= gp.Var)
match_col_df = np.ndarray(shape=(NCOL), dtype= gp.Var)
for j in COL:
    match_col_df_u[j] = m.addVar(lb=-NROW, ub=NROW, name="Match_C%d_u" %j)
    match_col_df[j] = m.addVar(lb=0, ub=NROW, vtype=GRB.INTEGER, name="Match_C%d" %j )

# define variables relate to df
final_df_b = m.addVar(lb=1, vtype=GRB.INTEGER, name="Final_b")
final_df_r = m.addVar(lb=1, vtype=GRB.INTEGER, name="Final_r")
match_df = m.addVar(lb=1, vtype=GRB.INTEGER, name="Match")
obj = m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

# create column-wise indicator variables to encode MC rules
class ColumnEncode(object):  
    def __init__(self, a: gp.Var, b:gp.Var, c:gp.Var):
        self.u = a
        self.x = b
        self.y = c

S_col = np.ndarray(shape = (total_r, NCOL), dtype = ColumnEncode)
M_col = np.ndarray(shape = (total_r, NCOL), dtype = ColumnEncode)

for r in range(total_r):
    for j in COL:
        S_col[r, j] = ColumnEncode(
            m.addVar(vtype=GRB.BINARY, name= "R%dSB_C%d_u" %(r,j)),
            m.addVar(vtype=GRB.BINARY, name= "R%dSB_C%d_x" %(r,j)),
            m.addVar(vtype=GRB.BINARY, name= "R%dSB_C%d_y" %(r,j)),
        )

        M_col[r, j] = ColumnEncode(
            m.addVar(vtype=GRB.BINARY, name= "R%dMC_C%d_u" %(r,j)),
            m.addVar(vtype=GRB.BINARY, name= "R%dMC_C%d_x" %(r,j)),
            m.addVar(vtype=GRB.BINARY, name= "R%dMC_C%d_y" %(r,j)),
        )

# add constraints for column encoding
for r in range(total_r):
    for j in COL:
        S_col[r, j].u == gp.max_(get_col(S[r,:,j], 'w').tolist())   # if one input cell is white, the output column is marked by u = 1 (unknown)
        S_col[r, j].x == gp.min_(get_col(S[r,:,j], 'b').tolist())   # if all cells are known for fwd computation, marked by x = 1
        S_col[r, j].y == gp.min_(get_col(S[r,:,j], 'r').tolist())   # if all cells are known for fwd computation, marked by y = 1

        M_col[r, j].u == gp.max_(get_col(M[r,:,j], 'w').tolist())   # if one input cell is white, the output column is marked by u = 1 (unknown)
        M_col[r, j].x == gp.min_(get_col(M[r,:,j], 'b').tolist())   # if all cells are known for fwd computation, marked by x = 1
        M_col[r, j].y == gp.min_(get_col(M[r,:,j], 'r').tolist())   # if all cells are known for fwd computation, marked by y = 1

# generate MC rule, calculate the consumed df for both fwd and bwd
def gen_MC_rule(input: np.ndarray, input_col: ColumnEncode, output: np.ndarray, fwd: gp.Var, bwd: gp.Var):
    m.addConstr(NROW * input_col.u + gp.quicksum(get_col(output, 'b')) <= NROW)
    m.addConstr(gp.quicksum(get_col(input, 'b')) + gp.quicksum(get_col(output, 'b')) - NBRANCH*input_col.x <= 2*NROW - NBRANCH)
    m.addConstr(gp.quicksum(get_col(input, 'b')) + gp.quicksum(get_col(output, 'b')) - 2*NROW*input_col.x >= 0)

    m.addConstr(NROW * input_col.u + gp.quicksum(get_col(output, 'r')) <= NROW)
    m.addConstr(gp.quicksum(get_col(input, 'r')) + gp.quicksum(get_col(output, 'r')) - NBRANCH*input_col.y <= 2*NROW - NBRANCH)
    m.addConstr(gp.quicksum(get_col(input, 'r')) + gp.quicksum(get_col(output, 'r')) - 2*NROW*input_col.y >= 0)

    m.addConstr(gp.quicksum(get_col(output, 'b')) - NROW * input_col.x - bwd == 0)
    m.addConstr(gp.quicksum(get_col(output, 'r')) - NROW * input_col.y - fwd == 0)

# generate match rule, calculate the degree of matching of each column
def gen_match_rule(input: np.ndarray, output: np.ndarray, match_u: gp.Var, match: gp.Var):
    m.addConstr(match_u == 
        gp.quicksum(get_col(input, 'b')) + gp.quicksum(get_col(input, 'r')) - gp.quicksum(get_col(input, 'g')) +
        gp.quicksum(get_col(output, 'b')) + gp.quicksum(get_col(output, 'r')) - gp.quicksum(get_col(output, 'g')) - NROW)
    m.addConstr(match == gp.max_(match_u, 0))

# set objective function: argmax obj = min{final_df_b, final_df_r, match_df}
def gen_objective():
    m.addConstr(final_df_b - gp.quicksum(start_df_b.flatten()) + gp.quicksum(consumed_fwd.flatten()) == 0)
    m.addConstr(final_df_r - gp.quicksum(start_df_r.flatten()) + gp.quicksum(consumed_bwd.flatten()) == 0)
    m.addConstr(match_df - gp.quicksum(match_col_df.flatten()) == 0)
    m.addConstr(obj - final_df_b <= 0)
    m.addConstr(obj - final_df_r <= 0)
    m.addConstr(obj - match_df <= 0)
    m.setObjective(obj, GRB.MAXIMIZE)

m.update()
# formulate the attack
def formulate():
    # start round: no unknowns
    for i in ROW:
        for j in COL:
            m.addConstr(S[start_r, i, j].b + S[start_r, i, j].r >= 1)
            m.addConstr(start_df_b[i, j] + S[start_r, i, j].r == 1)
            m.addConstr(start_df_r[i, j] + S[start_r, i, j].b == 1)
    
    # intermediate rounds: 
    for r in range(1, total_r - 1):
        nr = (r+1) % total_r
        if r in fwd:
            for j in range(NCOL):
                gen_MC_rule(input= M[r,:,j], input_col= M_col[r,j], output= S[nr,:,j], fwd= consumed_fwd[r,j], bwd= consumed_bwd[r,j])
        elif r in bwd:
            for j in range(NCOL):
                gen_MC_rule(input= S[nr,:,j], input_col= S_col[nr,j], output= M[r,:,j], fwd= consumed_fwd[r,j], bwd= consumed_bwd[r,j])
    
    # last round:
    lr = total_r - 1
    for j in COL:
        m.addConstr(consumed_fwd[lr, j] == 0)
        m.addConstr(consumed_bwd[lr, j] == 0)
        # jump the MC for last round
        for i in ROW:
            m.addConstr(M[lr, i, j].b - S[0, i, j].b == 0)
            m.addConstr(M[lr, i, j].r - S[0, i, j].r == 0)

    # match round:
    mid = match_r
    nmid = (mid + 1) % total_r
    for j in range(NCOL):
        gen_match_rule(input= M[mid,:,j], output= S[nmid,:,j], match_u = match_col_df_u[j], match= match_col_df[j])
        m.addConstr(consumed_fwd[mid, j] == 0)
        m.addConstr(consumed_bwd[mid, j] == 0)
    
    gen_objective()
    m.update()
    m.write(m.modelName + '.lp' )

def writeSol():
    if m.SolCount > 0:
        if m.getParamInfo(GRB.Param.PoolSearchMode)[2] > 0:
            gv = m.getVars()
            names = m.getAttr('VarName', gv)
            for i in range(m.SolCount):
                m.params.SolutionNumber = i
                xn = m.getAttr('Xn', gv)
                lines = ["{} {}".format(v1, v2) for v1, v2 in zip(names, xn)]
                with open('{}_{}.sol'.format(m.modelName, i), 'w') as f:
                    f.write("# Solution for model {}\n".format(m.modelName))
                    f.write("# Objective value = {}\n".format(m.PoolObjVal))
                    f.write("\n".join(lines))
        else:
            m.write(m.modelName + '.sol')
    else:
        print('infeasible')

formulate()
m.optimize()
writeSol()

def drawSol(total_r: int, ini_r: int, mat_r: int, F_r: list, B_r: list, outfile=None):
    if outfile == None:
        outfile = m.modelName + '.sol'
    solFile = open(outfile, 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))

    SB_x_v = np.ndarray(shape=(total_r, NROW, NCOL), dtype='int')
    SB_y_v = np.ndarray(shape=(total_r, NROW, NCOL), dtype='int')
    MC_x_v = np.ndarray(shape=(total_r, NROW, NCOL), dtype='int')
    MC_y_v = np.ndarray(shape=(total_r, NROW, NCOL), dtype='int')
    DoF_init_BL_v = np.ndarray(shape=(NROW, NCOL), dtype='int')
    DoF_init_RD_v = np.ndarray(shape=(NROW, NCOL), dtype='int')
    CD_x_v = np.ndarray(shape=(total_r, NCOL), dtype='int')
    CD_y_v = np.ndarray(shape=(total_r, NCOL), dtype='int')
    for ri in range(total_r):
        for i in range(NROW):
            for j in range(NCOL):
                SB_x_v[ri, i, j] = Sol["R%d[%d,%d]_b" %(ri,i,j)]
                SB_y_v[ri, i, j] = Sol["R%d[%d,%d]_r" %(ri,i,j)]
    for ri in range(total_r):
        for i in range(NROW):
            for j in range(NCOL):
                MC_x_v[ri, i, j] = SB_x_v[ri, i, (j + i)%NCOL]
                MC_y_v[ri, i, j] = SB_y_v[ri, i, (j + i)%NCOL]
    for i in range(NROW):
        for j in range(NCOL):
            DoF_init_BL_v[i, j] = Sol["Start[%d,%d]_b" %(i,j)]
            DoF_init_RD_v[i, j] = Sol["Start[%d,%d]_r" %(i,j)]
    for ri in range(total_r):
        for j in range(NCOL):
            CD_x_v[ri, j] = Sol["R%dC%d_fwd" %(ri,j)]
            CD_y_v[ri, j] = Sol["R%dC%d_bwd" %(ri,j)]
    DoF_BL_v = Sol["Final_b"]
    DoF_RD_v = Sol["Final_r"]
    DoM_v = Sol["Match"]

    CM = np.ndarray(shape=(2, 2),dtype='object')
    CM[0, 0] = '\\fill[\\UW]'
    CM[0, 1] = '\\fill[\\BW]'
    CM[1, 0] = '\\fill[\\FW]'
    CM[1, 1] = '\\fill[\\CW]'
    if NROW == 4:
        HO = NROW
        WO = NCOL
    else:
        HO = NROW
        WO = NCOL // 2
    ini_d1 = 0
    ini_d2 = 0
    for i in range(NROW):
        for j in range(NCOL):
            ini_d1 += DoF_init_BL_v[i,j]
            ini_d2 += DoF_init_RD_v[i,j]
    fid = open(outfile + '.tex', 'w')
    fid.write(
        '\\documentclass{standalone}' + '\n'
        '\\usepackage[usenames,dvipsnames]{xcolor}' + '\n'
        '\\usepackage{amsmath,amssymb,mathtools}' + '\n'
        '\\usepackage{tikz,calc,pgffor}' + '\n'
        '\\usepackage{xspace}' + '\n'
        '\\usetikzlibrary{crypto.symbols,patterns,calc}' + '\n'
        '\\tikzset{shadows=no}' + '\n'
        '\\input{macro}' + '\n')
    fid.write('\n\n')
    fid.write(
        '\\begin{document}' + '\n' +
        '\\begin{tikzpicture}[scale=0.2, every node/.style={font=\\boldmath\\bf}]' + '\n'
	    '\\everymath{\\scriptstyle}' + '\n'
	    '\\tikzset{edge/.style=->, >=stealth, arrow head=8pt, thick};' + '\n')
    fid.write('\n\n')

    for r in range(total_r):
        CD_BL = 0
        CD_RD = 0
        for i in range(NCOL):
            CD_BL += CD_x_v[r, i]
            CD_RD += CD_y_v[r, i]
        O = 0
        ## SB
        fid.write('\\begin{scope}[yshift =' + str(- r * (NROW + HO))+' cm, xshift =' + str(O * (NCOL + WO))+' cm]'+'\n')
        for i in range(NROW):
            row = NROW - 1 - i
            for j in range(NCOL):
                col = j
                fid.write(CM[SB_x_v[r,i,j], SB_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCOL) + ',' + str(NROW) + ');' + '\n')
        for i in range(1, NROW):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCOL) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCOL):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NROW) + ');' + '\n')
        fid.write('\\path (' + str(NCOL//2) + ',' + str(NROW + 0.5) + ') node {\\scriptsize$\\SB^' + str(r) + '$};'+'\n')
        if r in B_r:
            fid.write('\\draw[edge, <-] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
        else:
            fid.write('\\draw[edge, ->] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
        if r == ini_r:
            fid.write('\\path (' + str(NCOL//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_d1) + '~\\DoFF,~+' + str(ini_d2) + '~\\DoFB)$};'+'\n')
            fid.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        O = O + 1
        ## MC
        fid.write('\\begin{scope}[yshift =' + str(- r * (NROW + HO))+' cm, xshift =' +str(O * (NCOL + WO))+' cm]'+'\n')
        for i in range(NROW):
            row = NROW - 1 - i
            for j in range(NCOL):
                col = j
                fid.write(CM[MC_x_v[r,i,j], MC_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCOL) + ',' + str(NROW) + ');' + '\n')
        for i in range(1, NROW):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCOL) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCOL):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NROW) + ');' + '\n')
        fid.write('\\path (' + str(NCOL//2) + ',' + str(NROW + 0.5) + ') node {\\scriptsize$\\MC^' + str(r) + '$};'+'\n')
        op = 'MC'
        if r == total_r - 1:
            op = 'I'
        if r in B_r:
            fid.write('\\draw[edge, <-] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
        if r in F_r:
            fid.write('\\draw[edge, ->] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
        if r == mat_r:
            fid.write('\\draw[edge, -] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
            fid.write('\\draw[edge, ->] (' + str(NCOL) + ',' + str(NROW//2) + ') --  +(' + str(WO//2) + ',' + '0);' + '\n')
            fid.write('\\draw[edge, ->] (' + str(NCOL + WO) + ',' + str(NROW//2) + ') --  +(' + str(-WO//2) + ',' + '0);' + '\n')
    
            fid.write('\\path (' + str(NCOL + WO//2) + ',' + str(-0.8) + ') node {\\scriptsize Match};' + '\n')
            fid.write('\\path (' + str(-2) + ',' + str(0.1) + ') node {\\scriptsize$\\EndFwd$};' + '\n')
            fid.write('\\path (' + str(NCOL + WO + NCOL + 2) + ',' + str(0.1) + ') node {\\scriptsize$\\EndBwd$};' + '\n')
        else:
            fid.write('\\path (' + str((NCOL + WO) - WO//2) + ',' + str(-0.8) + ') node {\\scriptsize$ (-' + str(CD_BL) + '~\\DoFF,~-' + str(CD_RD) + '~\\DoFB)$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        O = O + 1
        ## SB r+1
        fid.write('\\begin{scope}[yshift =' + str(- r * (NROW + HO))+' cm, xshift =' +str(O * (NCOL + WO))+' cm]'+'\n')
        for i in range(NROW):
            row = NROW - 1 - i
            for j in range(NCOL):
                col = j
                fid.write(CM[SB_x_v[(r+1)%total_r,i,j], SB_y_v[(r+1)%total_r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCOL) + ',' + str(NROW) + ');' + '\n')
        for i in range(1, NROW):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCOL) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCOL):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NROW) + ');' + '\n')
        fid.write('\\path (' + str(NCOL//2) + ',' + str(NROW + 0.5) + ') node {\\scriptsize$\\SB^' + str((r+1)%total_r) + '$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')
    ## Final
    fid.write('\\begin{scope}[yshift =' + str(- total_r * (NROW + HO) + HO)+' cm, xshift =' +str(2 * (NCOL + WO))+' cm]'+'\n')
    fid.write(
        '\\node[draw, thick, rectangle, text width=6.5cm, label={[shift={(-2.8,-0)}]\\footnotesize Config}] at (-7, 0) {' + '\n'
	    '{\\footnotesize' + '\n'
	    '$\\bullet~(\\varInitBL,~\\varInitRD)~=~(+' + str(ini_d1) + '~\\DoFF,~+' + str(ini_d2) + '~\\DoFB)~$' + '\\\ \n'
	    '$\\bullet~(\\varDoFBL,~\\varDoFRD,~\\varDoM)~=~(+' + 
        str(int(DoF_BL_v)) + '~\\DoFF,~+' + 
        str(int(DoF_RD_v)) + '~\\DoFB,~+' + 
        str(int(DoM_v )) + '~\\DoM)$' + '\n'
	    '}' + '\n'
	    '};' + '\n'
        )
    fid.write('\n'+'\\end{scope}'+'\n')
    fid.write('\n\n')
    fid.write('\\end{tikzpicture}'+'\n\n'+'\\end{document}')
    fid.close()
    from os import system
    system("pdflatex -output-directory='./' ./" + outfile + ".tex") 

#fnp = m.modelName + '.sol'
#drawSol(7, 4, 1, fwd, bwd, fnp)