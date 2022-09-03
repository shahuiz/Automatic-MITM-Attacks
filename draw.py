total_r = 8
NROW = 8
NCOL = 8
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
                SB_x_v[ri, i, j] = Sol["SB_x" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
                SB_y_v[ri, i, j] = Sol["SB_y" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
    for ri in range(total_r):
        for i in range(NROW):
            for j in range(NCOL):
                MC_x_v[ri, i, j] = SB_x_v[ri, i, (j + i)%NCOL]
                MC_y_v[ri, i, j] = SB_y_v[ri, i, (j + i)%NCOL]
    for i in range(NROW):
        for j in range(NCOL):
            DoF_init_BL_v[i, j] = Sol["DoF_init_BL" + ('[%d,%d]' % (i,j))]
            DoF_init_RD_v[i, j] = Sol["DoF_init_RD" + ('[%d,%d]' % (i,j))]
    for ri in range(total_r):
        for j in range(NCOL):
            CD_x_v[ri, j] = Sol["CD_x_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri)]
            CD_y_v[ri, j] = Sol["CD_y_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri)]
    DoF_BL_v = Sol["DoF_BL"]
    DoF_RD_v = Sol["DoF_RD"]
    DoM_v = Sol["DoM"]

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