from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def writeSol(m: gp.Model):
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


def drawSol(TR, NRow, NCol, outfile=None):

    solFile = open(outfile, 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
    SB_x_v = np.ndarray(shape=(TR, NRow, NCol), dtype='int')
    SB_y_v = np.ndarray(shape=(TR, NRow, NCol), dtype='int')
    MC_x_v = np.ndarray(shape=(TR, NRow, NCol), dtype='int')
    MC_y_v = np.ndarray(shape=(TR, NRow, NCol), dtype='int')
    AK_x_v = np.ndarray(shape=(TR + 1, NRow, NCol), dtype='int')
    AK_y_v = np.ndarray(shape=(TR + 1, NRow, NCol), dtype='int')
    AT_x_v = np.ndarray(shape=(NRow, NCol), dtype='int')
    AT_y_v = np.ndarray(shape=(NRow, NCol), dtype='int')
    DoF_init_BL_v = np.ndarray(shape=(NRow, NCol), dtype='int')
    DoF_init_RD_v = np.ndarray(shape=(NRow, NCol), dtype='int')
    CD_x_v = np.ndarray(shape=(TR, NCol), dtype='int')
    CD_y_v = np.ndarray(shape=(TR, NCol), dtype='int')
    CD_ak_x_v = np.ndarray(shape=(self.TR + 1, NRow, NCol), dtype='object')
    CD_ak_y_v = np.ndarray(shape=(self.TR + 1, NRow, NCol), dtype='object')

    # RK
    RK_x_v = np.ndarray(shape=(self.TR + 1, NRow, NCol), dtype='int')
    RK_y_v = np.ndarray(shape=(self.TR + 1, NRow, NCol), dtype='int')
    DoF_init_rk_BL_v = np.ndarray(shape=(NRow, NCol), dtype='int')
    DoF_init_rk_RD_v = np.ndarray(shape=(NRow, NCol), dtype='int')
    CD_rk_x_v = np.ndarray(shape=(self.TR + 1, NRow, NCol), dtype='int')
    CD_rk_y_v = np.ndarray(shape=(self.TR + 1, NRow, NCol), dtype='int')

    for ri in range(self.TR):
        for i in range(NRow):
            for j in range(NCol):
                SB_x_v[ri, i, j] = Sol["SB_x" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
                SB_y_v[ri, i, j] = Sol["SB_y" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
    for ri in range(self.TR):
        for i in range(NRow):
            for j in range(NCol):
                MC_x_v[ri, i, j] = SB_x_v[ri, i, (j + i)%NCol]
                MC_y_v[ri, i, j] = SB_y_v[ri, i, (j + i)%NCol]
    for i in range(NRow):
        for j in range(NCol):
            DoF_init_BL_v[i, j] = Sol["DoF_init_BL" + ('[%d,%d]' % (i,j))]
            DoF_init_RD_v[i, j] = Sol["DoF_init_RD" + ('[%d,%d]' % (i,j))]
            AT_x_v[i, j] = Sol["AT_x" + ('[%d,%d]' % (i,j))]
            AT_y_v[i, j] = Sol["AT_y" + ('[%d,%d]' % (i,j))]

    for ri in range(self.TR):
        for j in range(NCol):
            CD_x_v[ri, j] = Sol["CD_x_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri)]
            CD_y_v[ri, j] = Sol["CD_y_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri)]
    DoF_ALL_BL_v = Sol["DoF_ALL_BL"]
    DoF_ALL_RD_v = Sol["DoF_ALL_RD"]
    DoM_v = Sol["DoM"]

    # RK
    for ri in range(self.TR + 1):
        for i in range(NRow):
            for j in range(NCol):
                RK_x_v[ri, i, j] = Sol["RK_x" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
                RK_y_v[ri, i, j] = Sol["RK_y" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
                AK_x_v[ri, i, j] = Sol["AK_x" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
                AK_y_v[ri, i, j] = Sol["AK_y" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
    for i in range(NRow):
        for j in range(NCol):
            DoF_init_rk_BL_v[i, j] = Sol["DoF_init_rk_BL" + ('[%d,%d]' % (i,j))]
            DoF_init_rk_RD_v[i, j] = Sol["DoF_init_rk_RD" + ('[%d,%d]' % (i,j))]
    for ri in range(self.TR + 1):
        for i in range(NRow):
            for j in range(NCol):
                CD_rk_x_v[ri, i, j] = Sol["CD_rk_x" + ('[%d,%d]' % (i,j)) + "_r" + ('[%d]' % ri)]
                CD_rk_y_v[ri, i, j] = Sol["CD_rk_y" + ('[%d,%d]' % (i,j)) + "_r" + ('[%d]' % ri)]
                CD_ak_x_v[ri, i, j] = Sol["CD_ak_x" + ('[%d,%d]' % (i,j)) + "_r" + ('[%d]' % ri)]
                CD_ak_y_v[ri, i, j] = Sol["CD_ak_y" + ('[%d,%d]' % (i,j)) + "_r" + ('[%d]' % ri)]

    DoF_rk_BL_v = Sol["DoF_rk_BL"]
    DoF_rk_RD_v = Sol["DoF_rk_RD"]

    #
    CM = np.ndarray(shape=(2, 2),dtype='object')
    CM[0, 0] = '\\fill[\\UW]'
    CM[0, 1] = '\\fill[\\BW]'
    CM[1, 0] = '\\fill[\\FW]'
    CM[1, 1] = '\\fill[\\CW]'
    if NRow == 4:
        HO = NRow
        WO = NCol
    else:
        HO = NRow
        WO = NCol // 2
    ini_d1 = 0
    ini_d2 = 0
    ini_rk_d1 = 0
    ini_rk_d2 = 0

    for i in range(NRow):
        for j in range(NCol):
            ini_d1 += DoF_init_BL_v[i,j]
            ini_d2 += DoF_init_RD_v[i,j]
            ini_rk_d1 += DoF_init_rk_BL_v[i,j]
            ini_rk_d2 += DoF_init_rk_RD_v[i,j]

    fid = open(outfile + '.tex', 'w')
    fid.write(
        '\\documentclass{standalone}' + '\n'
        '\\usepackage[usenames,dvipsnames]{xcolor}' + '\n'
        '\\usepackage{amsmath,amssymb,mathtools}' + '\n'
        '\\usepackage{tikz,calc,pgffor}' + '\n'
        '\\usepackage{xspace}' + '\n'
        '\\usetikzlibrary{crypto.symbols,patterns,calc,positioning,scopes,shapes,snakes}' + '\n'
        '\\tikzset{shadows=no}' + '\n'
        '\\input{macro}' + '\n')
    fid.write('\n\n')
    fid.write(
        '\\begin{document}' + '\n' +
        '\\begin{tikzpicture}[scale=0.2, '
        'every node/.style={font=\\boldmath\\bf}]' + '\n'
        '\\everymath{\\scriptstyle}' + '\n'
        '\\tikzset{edge/.style=->, >=stealth, arrow head=8pt, thick};' + '\n'
    )
    fid.write('\n\n')
    #
    for r in range(self.TR):
        CD_BL = 0
        CD_RD = 0
        for i in range(NCol):
            CD_BL += CD_x_v[r, i]
            CD_RD += CD_y_v[r, i]
        O = 0
        ## SB
        fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+' cm, xshift =' + str(O * (NCol + WO))+' cm]'+'\n')
        for i in range(NRow):
            row = NRow - 1 - i
            for j in range(NCol):
                col = j
                fid.write(CM[SB_x_v[r,i,j], SB_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
        for i in range(1, NRow):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCol):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
        fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\SB^' + str(r) + '$};'+'\n')
        if r in self.B_r:
            fid.write('\\draw[edge, <-] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
        else:
            fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
        if r == self.ini_r:
            fid.write('\\path (' + str(NCol//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_d1) + '~\\DoFF,~+' + str(ini_d2) + '~\\DoFB)$};'+'\n')
            fid.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        O = O + 1
        ## MC
        fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+' cm, xshift =' +str(O * (NCol + WO))+' cm]'+'\n')
        for i in range(NRow):
            row = NRow - 1 - i
            for j in range(NCol):
                col = j
                fid.write(CM[MC_x_v[r,i,j], MC_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
        for i in range(1, NRow):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCol):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
        fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\MC^' + str(r) + '$};'+'\n')
        op = 'MC'
        if r == self.TR - 1 and WLastMC == 0:
            op = 'I'
        if r in self.B_r:
            fid.write('\\draw[edge, <-] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
        if r in self.F_r:
            fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
        if r == self.mat_r:
            if self.mat_r != self.TR - 1:
                fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) +
                        ') -- +(' + str(WO) + ',' + '0);' + '\n')

                fid.write('\\draw[edge, -] (' + str(NCol+ 2 * WO) + ',' + str(NRow//2) +
                        ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
                fid.write('\\draw[edge, ->] (' + str(NCol+ 2 * WO) + ',' + str(NRow//2) +
                        ') --  +(' + str(WO//2) + ',' + '0);' + '\n')
                fid.write('\\draw[edge, ->] (' + str(NCol + WO+ 2 * WO) + ',' +
                        str(NRow//2) + ') --  +(' + str(-WO//2) + ',' + '0);' + '\n')
                fid.write('\\node[scale=.8, xor] at ' +
                        '(' + str(NCol + WO//2) + ', ' + str(NRow//2) + ') {};' + '\n')
                fid.write('\\path (' + str(NCol + WO//2 + 2 * WO) + ',' + str(-0.8) + ') node {\\scriptsize Match};' + '\n')
            else:
                fid.write('\\draw[edge, -] (' + str(NCol) + ',' + str(NRow//2) +
                        ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
                fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) +
                        ') --  +(' + str(WO//2) + ',' + '0);' + '\n')
                fid.write('\\draw[edge, ->] (' + str(NCol + WO) + ',' +
                        str(NRow//2) + ') --  +(' + str(-WO//2) + ',' + '0);' + '\n')
                fid.write('\\path (' + str(NCol + WO//2) + ',' + str(-0.8) + ') node {\\scriptsize Match};' + '\n')
            #  fid.write('\\path (' + str(-2) + ',' + str(0.1) + ') node {\\scriptsize$\\EndFwd$};' + '\n')
            #  fid.write('\\path (' + str(NCol + WO + NCol + 2) + ',' + str(0.1) + ') node {\\scriptsize$\\EndBwd$};' + '\n')
        else:
            fid.write('\\path (' + str((NCol + WO) - WO//2) + ',' + str(-0.8) + ') node {\\scriptsize$ (-' + str(CD_BL) + '~\\DoFF,~-' + str(CD_RD) + '~\\DoFB)$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        O = O + 1
        ## AK r
        fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+
                ' cm, xshift =' +str(O * (NCol + WO))+' cm]'+'\n')
        if r not in self.B_r:
            for i in range(NRow):
                row = NRow - 1 - i
                for j in range(NCol):
                    col = j
                    fid.write(CM[AK_x_v[(r+1)%(self.TR+1),i,j], AK_y_v[(r+1)%(self.TR+1),i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
            fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
            for i in range(1, NRow):
                fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
            for i in range(1, NCol):
                fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
        else:
            fid.write('\\draw[pattern =  north east lines] (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')

        fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) +
                ') node {\\scriptsize$\\AK^' + str((r)%self.TR) + '$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        O = O + 1
        ## SB r+1
        if r < self.TR - 1:
            fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+
                    ' cm, xshift =' +str(O * (NCol + WO))+' cm]'+'\n')
            for i in range(NRow):
                row = NRow - 1 - i
                for j in range(NCol):
                    col = j
                    fid.write(CM[SB_x_v[(r+1)%self.TR,i,j], SB_y_v[(r+1)%self.TR,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
            fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
            for i in range(1, NRow):
                fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
            for i in range(1, NCol):
                fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
            fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\SB^' + str((r+1)%self.TR) + '$};'+'\n')
        else:
            fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+
                    ' cm, xshift =' +str(O * (NCol + WO))+' cm]'+'\n')
            for i in range(NRow):
                row = NRow - 1 - i
                for j in range(NCol):
                    col = j
                    fid.write(CM[AT_x_v[i,j], AT_y_v[i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
            fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
            for i in range(1, NRow):
                fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
            for i in range(1, NCol):
                fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
            fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\AT$};'+'\n')
            fid.write('\\draw[scale=1, edge, -] (' +
                    str(NCol//2) + ',' + str(0) + ') -- +(' +
                    '0, ' + str(-NRow) + ');' + '\n')
            fid.write('\\draw[scale=1, edge, -] (' +
                    str(NCol//2) + ',' + str(-NRow) + ') -- +(' +
                    str(-NCol * 8) + ', ' + str(0) + ');' + '\n')
            fid.write('\\draw[scale=1, edge, -] (' +
                    str(-NCol * 8 + NCol//2) + ',' + str(-NRow) + ') -- +(' +
                    str(0) + ', ' + str((self.TR + 1) * NRow * 2 - NRow // 2) + ');' + '\n')
            fid.write('\\draw[scale=1, edge, ->] (' +
                    str(-NCol * 8 + NCol//2) + ',' + str((self.TR + 1) * NRow * 2 - NRow // 2 - NRow) + ') -- +(' +
                    str(2 * NCol - 0.7) + ', ' + str(0) + ');' + '\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

    ## Final
    fid.write('\\begin{scope}[yshift =' + str(- (self.TR + 1) * (NRow + HO) + HO)+' cm, xshift =' +str(2 * (NCol + WO))+' cm]'+'\n')
    fid.write(
        '\\node[draw, thick, rectangle, text width=8cm, label={[shift={(-2.8,-0)}]\\footnotesize Config}] at (-7, 0) {' + '\n'
        '{\\footnotesize' + '\n'
        '$\\bullet~(\\varInitBL,~\\varInitRD)~=~(+(' + str(ini_d1) +
        ' + ' + str(ini_rk_d1) + ') = ' + str(ini_d1 + ini_rk_d1) +
        '~\\DoFF,~+(' + str(ini_d2) + ' + ' + str(ini_rk_d2) + ') = ' +
        str(ini_d2 + ini_rk_d2) + '~\\DoFB)~$' + '\\\ \n'
        '$\\bullet~(\\varDoFBL,~\\varDoFRD,~\\varDoM)~=~(+' +
        str(int(DoF_ALL_BL_v)) + '~\\DoFF,~+' +
        str(int(DoF_ALL_RD_v)) + '~\\DoFB,~+' +
        str(int(DoM_v )) + '~\\DoM)$' + '\n'
        '}' + '\n'
        '};' + '\n'
        )
    fid.write('\n'+'\\end{scope}'+'\n')
    fid.write('\n\n')

    #RK
    for r in range(self.TR + 1):
        CD_rk_BL = 0
        CD_rk_RD = 0
        CD_ak_BL = 0
        CD_ak_RD = 0
        for i in range(NRow):
            for j in range(NCol):
                CD_rk_BL += CD_rk_x_v[r, i, j]
                CD_rk_RD += CD_rk_y_v[r, i, j]
                CD_ak_BL += CD_ak_x_v[r, i, j]
                CD_ak_RD += CD_ak_y_v[r, i, j]

        O = 4.5
        ## SB
        fid.write('\\begin{scope}[yshift =' + str(0 - (r - 1) * (NRow + HO))+' cm, xshift =' +
                str(O * (NCol + WO))+' cm]'+'\n')
        for i in range(NRow):
            row = NRow - 1 - i
            for j in range(NCol):
                col = j
                fid.write(CM[RK_x_v[r,i,j], RK_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
        for i in range(1, NRow):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCol):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
        fid.write('\\path (' + str(-1) + ',' + str(NRow//2) +
                    ') node {\\scriptsize$k^{' + str(r-1) + '}$};'+'\n')
        if r < self.TR:
            # Col_3
            fid.write('\\draw[scale=1, thin, edge, ->] (' + str(3.5) + ',' + str(0) +
                    ') -- +(' +
                    '0, ' + str(-WO*7.5/10) + ');' + '\n')
            fid.write('\\node[scale=.3, thin, XOR] at ' +
                    '(3.5, ' + str(-WO*7.5/10-0.2) + ') {};' + '\n')
            fid.write('\\draw[scale=1, thin, edge, ->] (' +
                    str(3.5) + ',' + str(-WO*7.5/10-.4) + ') -- +(' +
                    '0, ' + str(-WO*2.5/10+.4) + ');' + '\n')
            fid.write('\\draw[scale=1, thin, edge, <-] (' +
                    str(3.3) + ',' + str(-WO*7.5/10-.4+0.2) + ') -- +(' +
                    '-.8, ' + str(0) + ');' + '\n')

            # Col_2
            fid.write('\\draw[scale=1, thin, edge, ->] (' + str(2.5) + ',' + str(0) +
                    ') -- +(' +
                    '0, ' + str(-WO*6/10) + ');' + '\n')
            fid.write('\\node[scale=.3, thin, XOR] at ' +
                    '(2.5, ' + str(-WO*6/10-0.2) + ') {};' + '\n')
            fid.write('\\draw[scale=1, thin, edge, ->] (' +
                    str(2.5) + ',' + str(-WO*6/10-.4) + ') -- +(' +
                    '0, ' + str(-WO*4/10+.4) + ');' + '\n')
            fid.write('\\draw[scale=1, thin, edge, <-] (' +
                    str(2.3) + ',' + str(-WO*6/10-.4+0.2) + ') -- +(' +
                    '-.8, ' + str(0) + ');' + '\n')

            # Col_1
            fid.write('\\draw[scale=1, thin, edge, ->] (' + str(1.5) + ',' + str(0) +
                    ') -- +(' +
                    '0, ' + str(-WO*4.5/10) + ');' + '\n')
            fid.write('\\node[scale=.3, thin, XOR] at ' +
                    '(1.5, ' + str(-WO*4.5/10-0.2) + ') {};' + '\n')
            fid.write('\\draw[scale=1, thin, edge, ->] (' +
                    str(1.5) + ',' + str(-WO*4.5/10-.4) + ') -- +(' +
                    '0, ' + str(-WO*5.5/10+.4) + ');' + '\n')
            fid.write('\\draw[scale=1, thin, edge, <-] (' +
                    str(1.3) + ',' + str(-WO*4.5/10-.4+0.2) + ') -- +(' +
                    '-.8, ' + str(0) + ');' + '\n')
            # Col_0
            fid.write('\\draw[scale=1, thin, edge, ->] (' + str(0.5) + ',' + str(0) +
                    ') -- +(' +
                    '0, ' + str(-WO*2/10) + ');' + '\n')
            fid.write('\\node[scale=.3, thin, XOR] at ' +
                    '(0.5, ' + str(-WO*2/10-0.2) + ') {};' + '\n')
            fid.write('\\draw[scale=1, thin, edge, ->] (' +
                    str(0.5) + ',' + str(-WO*2/10-.4) + ') -- +(' +
                    '0, ' + str(-WO*8/10+.4) + ');' + '\n')

            fid.write('\\node[scale=0.3] at (' +
                    str(NCol - 1.1) + ',' + str(-WO*2/10-0.2) +
                    ') {{$\\lll$}};' + '\n')
            fid.write('\\node[scale=0.3] at (' +
                    str(NCol - 1.1 - 1) + ',' + str(-WO*2/10-0.2) +
                    ') {S};' + '\n')
            fid.write('\\draw[scale=1, thin, edge, ->] (' +
                    str(NCol - 1.1 - 1 - 0.2) + ',' + str(-WO*2/10-0.2) + ') -- +(' +
                    '-1.0, ' + str(0) + ');' + '\n')
            fid.write('\\draw[scale=1, thin, edge, -] (' +
                    str(NCol - 0.5) + ',' + str(-WO*2/10-0.2) + ') -- +(' +
                    '-0.4, ' + str(0) + ');' + '\n')
            fid.write('\\draw[scale=1, thin, edge, -] (' +
                    str(NCol - 1.1 - 0.3) + ',' + str(-WO*2/10-0.2) + ') -- +(' +
                    '-0.5, ' + str(0) + ');' + '\n')
        else:
            pass

        if r == self.ini_rk:
            fid.write('\\path (' + str(NCol*2) + ',' + str(NRow//2 + 1) + ') node {\\scriptsize$(+' +
                    str(ini_rk_d1) + '~\\DoFF,$};'+'\n')
            fid.write('\\path (' + str(NCol*2) + ',' + str(NRow//2 - 1) + ') node {\\scriptsize$+' +
                    str(ini_rk_d2) + '~\\DoFB)$};'+'\n')
            fid.write('\\path (' + str(-2) + ',' + str(0.8) +
                    ') node {\\scriptsize$\\StKSA$};'+'\n')
        else:
            fid.write('\\path (' + str(NCol*2) + ',' + str(NRow//2 + 1) + ') node {\\scriptsize$(-' +
                    str(CD_rk_BL) + '~\\DoFF,$};'+'\n')
            fid.write('\\path (' + str(NCol*2) + ',' + str(NRow//2 - 1) + ') node {\\scriptsize$-' +
                    str(CD_rk_RD) + '~\\DoFB)$};'+'\n')

        Arrow_dir = "->"
        Arrow_dir_inv = "<-"
        if r - 1 in self.B_r:
            Arrow_dir = "<-"
            Arrow_dir_inv = "->"
        if 0 < r:
            # AK
            fid.write('\\draw[scale=1, edge, -] (' +
                    str(-2) + ',' + str(NRow//2) + ') -- +(' +
                    '-2, ' + str(0) + ');' + '\n')
            fid.write('\\draw[scale=1, edge, -] (' +
                    str(-4) + ',' + str(NRow//2) + ') -- +(' +
                    '-0, ' + str(NRow//2 + 1.5) + ');' + '\n')
            fid.write('\\draw[scale=1, edge, -] (' +
                    str(-4) + ',' + str(NRow + 1.5) + ') -- +(' +
                    '-10, ' + str(0) + ');' + '\n')
            if r == self.mat_r + 1 and self.mat_r != self.TR - 1:
                #  fid.write('\\draw[scale=1, edge, ' + Arrow_dir_inv + '] (' +
                #          str(-16) + ',' + str(NRow//2) + ') -- +(' +
                #          str(NRow) + ', ' + str(0) + ');' + '\n')
                fid.write('\\draw[scale=1, edge, -] (' +
                        str(-14) + ',' + str(NRow + 1.5) + ') -- +(' +
                        str(-2 * NCol) + ', ' + str(0) + ');' + '\n')
                fid.write('\\draw[scale=1, edge, ->] (' +
                        str(-14 -2 * NCol) + ',' + str(NRow + 1.5) + ') -- +(' +
                        str(0) + ', ' + str(-NRow//2-0.8) + ');' + '\n')

                fid.write('\\path (' + str((NCol + WO) - WO * 7.5) + ',' +
                        str(-0.8) + ') node {\\scriptsize$ (-' +
                        str(CD_ak_BL) + '~\\DoFF,~-' + str(CD_ak_RD) + '~\\DoFB)$};'+'\n')
            else:
                fid.write('\\path (' + str((NCol + WO) - WO * 5) + ',' +
                        str(-0.8) + ') node {\\scriptsize$ (-' +
                        str(CD_ak_BL) + '~\\DoFF,~-' + str(CD_ak_RD) + '~\\DoFB)$};'+'\n')

                fid.write('\\draw[scale=1, edge, ->] (' +
                        str(-14) + ',' + str(NRow + 1.5) + ') -- +(' +
                        '-0, ' + str(-NRow//2-0.8) + ');' + '\n')
                fid.write('\\node[scale=.8, xor] at ' +
                        '(-14, ' + str(NRow//2) + ') {};' + '\n')
                if r != self.mat_r + 1:
                    fid.write('\\draw[scale=1, edge, ' + Arrow_dir_inv + '] (' +
                            str(-12) + ',' + str(NRow//2) + ') -- +(' +
                            '-1.3, ' + str(0) + ');' + '\n')
                    fid.write('\\draw[scale=1, edge, ' + Arrow_dir + '] (' +
                            str(-16) + ',' + str(NRow//2) + ') -- +(' +
                            '1.3, ' + str(0) + ');' + '\n')
                    pass
                else:
                    fid.write('\\draw[scale=1, edge, ' + Arrow_dir + '] (' +
                            str(-12) + ',' + str(NRow//2) + ') -- +(' +
                            '-1.3, ' + str(0) + ');' + '\n')
                    fid.write('\\draw[scale=1, edge, ' + Arrow_dir_inv + '] (' +
                            str(-16) + ',' + str(NRow//2) + ') -- +(' +
                            '1.3, ' + str(0) + ');' + '\n')
        else:
            fid.write('\\draw[scale=1, edge, ->] (' +
                    str(-2.5) + ',' + str(NRow//2) + ') -- +(' +
                    str(-NCol * 8 + 1.2) + ', ' + str(0) + ');' + '\n')
            fid.write('\\node[scale=.8, xor] at ' +
                    '(' + str(-NCol * 8 - NCol//2) + ', ' + str(NRow//2) + ') {};' + '\n')
            fid.write('\\draw[scale=1, edge, ->] (' +
                    str(-NCol * 8 - NCol//2) + ',' + str(NRow//2) + ') -- +(' +
                    str(0) + ', ' + str(-NRow - 0.5) + ');' + '\n')

            fid.write('\\path (' + str((NCol + WO) - WO * 10.5) + ',' +
                    str(-0.8) + ') node {\\scriptsize$ (-' +
                    str(CD_ak_BL) + '~\\DoFF,~-' + str(CD_ak_RD) + '~\\DoFB)$};'+'\n')

        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

    fid.write('\\end{tikzpicture}'+'\n\n'+'\\end{document}')
    fid.close()
    system("pdflatex -output-directory=" + self.fpath + ' ' + outfile + ".tex")
