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

m = gp.Model('model_%dx%d_%dR_Start_r%d_Meet_r%d' % (NROW, NCOL, total_round, start_round, match_round))

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

    # define variables for columnwise encoding
    S_col_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_u').values()).reshape((total_r, NCOL))
    S_col_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_x').values()).reshape((total_r, NCOL))
    S_col_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='S_col_y').values()).reshape((total_r, NCOL))

    M_col_u = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_u').values()).reshape((total_r, NCOL))
    M_col_x = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_x').values()).reshape((total_r, NCOL))
    M_col_y = np.asarray(m.addVars(total_r, NCOL, vtype=GRB.BINARY, name='M_col_y').values()).reshape((total_r, NCOL))

    # define vars to track the start state
    start_b = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Start_b').values()).reshape((NROW, NCOL))
    start_r = np.asarray(m.addVars(NROW, NCOL, vtype=GRB.BINARY, name='Start_r').values()).reshape((NROW, NCOL))
    
    # define auxiliary vars tracking cost of df at MC operations
    cost_fwd = np.asarray(m.addVars(total_r, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Cost_fwd').values()).reshape((total_r, NCOL))
    cost_bwd = np.asarray(m.addVars(total_r, NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Cost_bwd').values()).reshape((total_r, NCOL))
    
    # define auxiliary vars for computations on degree of matching
    meet_signed = np.asarray(m.addVars(NCOL, lb=-NROW, ub=NROW, vtype=GRB.INTEGER, name='Meet_signed').values())
    meet = np.asarray(m.addVars(NCOL, lb=0, ub=NROW, vtype=GRB.INTEGER, name='Meet').values())
    
    m.update()
    return S_b, S_r, S_g, S_w, M_b, M_r, M_g, M_w, S_col_u, S_col_x, S_col_y, M_col_u, M_col_x, M_col_y, start_b, start_r, cost_fwd, cost_bwd, meet_signed, meet

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


#############################################################################
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

S_b, S_r, S_g, S_w, M_b, M_r, M_g, M_w, S_col_u, S_col_x, S_col_y, M_col_u, M_col_x, M_col_y, start_b, start_r, cost_fwd, cost_bwd, meet_signed, meet = def_var(total_round, m)
gen_encode_rule(m, total_round, S_b, S_r, S_g, S_w, M_b, M_r, M_g, M_w, S_col_u, S_col_x, S_col_y, M_col_u, M_col_x, M_col_y)

for r in range(total_round):
    print(r)
    nr = (r+1) % total_round
    if r == start_round:
        print('start', r)
        for i in ROW:
            for j in COL:
                m.addConstr(S_b[r, i, j] + S_r[r, i, j] >= 1)
                m.addConstr(start_b[i, j] + S_r[r, i, j] == 1)
                m.addConstr(start_r[i, j] + S_b[r, i, j] == 1)
    if r == match_round:
        print('mat', r)
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
                #continue
                gen_MC_rule(m, M_b[r,:,j], M_r[r,:,j], M_col_u[r,j], M_col_x[r,j], M_col_y[r,j], S_b[nr,:,j], S_r[nr,:,j], cost_fwd[r,j], cost_bwd[r,j])
                m.update()
                print(m)
        elif r in bwd:
            print('bwd', r)
            for j in COL:
                #continue
                gen_MC_rule(m, S_b[nr,:,j], S_r[nr,:,j], S_col_u[nr,j], S_col_x[nr,j], S_col_y[nr,j], M_b[r,:,j], M_r[r,:,j], cost_fwd[r,j], cost_bwd[r,j])

def writeSol():
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
            m.write('./runlog/' + m.modelName + '.sol')
    else:
        print('infeasible')

def drawSol(total_r: int, ini_r: int, mat_r: int, F_r: list, B_r: list, outfile=None):
    if outfile == None:
        outfile = './runlog/' + m.modelName + '.sol'
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
                SB_x_v[ri, i, j] = Sol["S_b[%d,%d,%d]" %(ri,i,j)]
                SB_y_v[ri, i, j] = Sol["S_r[%d,%d,%d]" %(ri,i,j)]
    
    for ri in range(total_r):
        for i in range(NROW):
            for j in range(NCOL):
                MC_x_v[ri, i, j] = SB_x_v[ri, i, (j + i)%NCOL]
                MC_y_v[ri, i, j] = SB_y_v[ri, i, (j + i)%NCOL]
    
    for i in range(NROW):
        for j in range(NCOL):
            DoF_init_BL_v[i, j] = Sol["Start_b[%d,%d]" %(i,j)]
            DoF_init_RD_v[i, j] = Sol["Start_r[%d,%d]" %(i,j)]
    for ri in range(total_r):
        for j in range(NCOL):
            CD_x_v[ri, j] = Sol["Cost_fwd[%d,%d]" %(ri,j)]
            CD_y_v[ri, j] = Sol["Cost_bwd[%d,%d]" %(ri,j)]
    
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


set_obj(m, start_b, start_r, cost_fwd, cost_bwd, meet)
m.optimize()
writeSol()
print(m)

fnp = './runlog/' + m.modelName + '.sol'
drawSol(7, 4, 1, fwd, bwd, fnp)