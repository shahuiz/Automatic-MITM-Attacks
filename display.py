# simple txt display of the solution
from io import TextIOWrapper
import gurobipy as gp
from gurobipy import GRB
from string import Template
import numpy as np
import re
import os
import math

# AES parameters
NROW = 4
NCOL = 4
NBYTE = 32
NGRID = NROW * NCOL
NBRANCH = NROW + 1
ROW = range(NROW)
COL = range(NCOL)
TAB = ' ' * 4

def displaySol(key_size:int, total_round:int, enc_start_round:int, match_round:int, key_start_round:int, model_name:str, sol_i:int, dir):
    def color(b,r):
        if b==1 and r==0:
            return 'b'
        if b==0 and r==1:
            return 'r'
        if b==1 and r==1:
            return 'g'
        if b==0 and r==0:
            return 'w'

    solFile = open(dir + model_name + '_sol_' + str(sol_i) + '.sol', 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
 
    if enc_start_round < match_round:
        fwd = list(range(enc_start_round, match_round))
        bwd = list(range(match_round + 1, total_round)) + list(range(0, enc_start_round))
    else:
        bwd = list(range(match_round + 1, enc_start_round))
        fwd = list(range(enc_start_round, total_round)) + list(range(0, match_round))

    Nb = NCOL
    Nk = key_size // NBYTE
    Nr = math.ceil((total_round + 1)*Nb / Nk)
        
    SB_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    SB_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fSB_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fSB_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bSB_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bSB_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    MC_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fMC_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fMC_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bMC_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bMC_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fAK_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fAK_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAK_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAK_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fKEY_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fKEY_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKEY_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKEY_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

    fKSch_x= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    fKSch_y= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKSch_x= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKSch_y= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)

    KSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    KSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    fKSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    fKSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    bKSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    bKSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    
    Key_cost_fwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    Key_cost_bwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    xor_cost_fwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    xor_cost_bwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    mc_cost_fwd = np.ndarray(shape=(total_round, NCOL), dtype=int)
    mc_cost_bwd = np.ndarray(shape=(total_round, NCOL), dtype=int)

    ini_enc_x = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_y = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_g = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_x = np.ndarray(shape=(NROW, Nk), dtype=int)
    ini_key_y = np.ndarray(shape=(NROW, Nk), dtype=int)
    ini_key_g = np.ndarray(shape=(NROW, Nk), dtype=int)
    
    Meet_fwd_x = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_fwd_y = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_bwd_x = np.ndarray(shape=(NROW, NCOL), dtype=int)
    Meet_bwd_y = np.ndarray(shape=(NROW, NCOL), dtype=int)
    meet =  np.ndarray(shape=(NCOL), dtype=int)
    meet_s =  np.ndarray(shape=(NCOL), dtype=int)

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                SB_x[r,i,j]=Sol["S_x[%d,%d,%d]" %(r,i,j)]
                SB_y[r,i,j]=Sol["S_y[%d,%d,%d]" %(r,i,j)]
                fSB_x[r,i,j]=Sol["fS_x[%d,%d,%d]" %(r,i,j)]
                fSB_y[r,i,j]=Sol["fS_y[%d,%d,%d]" %(r,i,j)]
                bSB_x[r,i,j]=Sol["bS_x[%d,%d,%d]" %(r,i,j)]
                bSB_y[r,i,j]=Sol["bS_y[%d,%d,%d]" %(r,i,j)]

                fAK_x[r,i,j]=Sol["fA_x[%d,%d,%d]" %(r,i,j)]
                fAK_y[r,i,j]=Sol["fA_y[%d,%d,%d]" %(r,i,j)]
                bAK_x[r,i,j]=Sol["bA_x[%d,%d,%d]" %(r,i,j)]
                bAK_y[r,i,j]=Sol["bA_y[%d,%d,%d]" %(r,i,j)]

                fMC_x[r,i,j]=Sol["fM_x[%d,%d,%d]" %(r,i,j)]
                fMC_y[r,i,j]=Sol["fM_y[%d,%d,%d]" %(r,i,j)]
                bMC_x[r,i,j]=Sol["bM_x[%d,%d,%d]" %(r,i,j)]
                bMC_y[r,i,j]=Sol["bM_y[%d,%d,%d]" %(r,i,j)]

    for ri in range(total_round):
        for i in ROW:
            for j in COL:
                MC_x[ri, i, j] = SB_x[ri, i, (j + i)%NCOL]
                MC_y[ri, i, j] = SB_y[ri, i, (j + i)%NCOL]

    
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_round):
        for j in COL:
            for i in ROW:
                fKEY_x[r,i,j] = Sol["fKS_x[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                fKEY_y[r,i,j] = Sol["fKS_y[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bKEY_x[r,i,j] = Sol["bKS_x[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bKEY_y[r,i,j] = Sol["bKS_y[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0
    
    for r in range(Nr):
        for i in ROW:
            for j in range(Nk):
                fKSch_x[r,i,j] = Sol["fKS_x[%d,%d,%d]" %(r,i,j)]
                fKSch_y[r,i,j] = Sol["fKS_y[%d,%d,%d]" %(r,i,j)]
                bKSch_x[r,i,j] = Sol["bKS_x[%d,%d,%d]" %(r,i,j)]
                bKSch_y[r,i,j] = Sol["bKS_y[%d,%d,%d]" %(r,i,j)]

                Key_cost_fwd[r,i,j] = Sol["true_key_cost_fwd[%d,%d,%d]" %(r,i,j)]
                Key_cost_bwd[r,i,j] = Sol["true_key_cost_bwd[%d,%d,%d]" %(r,i,j)]
    
    for r in range(Nr):
        for i in ROW:
            KSub_x[r,i] = Sol["Ksub_x[%d,%d]" %(r,i)]
            KSub_y[r,i] = Sol["Ksub_y[%d,%d]" %(r,i)]
            fKSub_x[r,i] = Sol["fKsub_x[%d,%d]" %(r,i)]
            fKSub_y[r,i] = Sol["fKsub_y[%d,%d]" %(r,i)]
            bKSub_x[r,i] = Sol["bKsub_x[%d,%d]" %(r,i)]
            bKSub_y[r,i] = Sol["bKsub_y[%d,%d]" %(r,i)]
    
    for r in range(total_round+1):
        for i in ROW:
            for j in COL:
                xor_cost_fwd[r,i,j] = Sol["XOR_Cost_fwd[%d,%d,%d]" %(r,i,j)]
                xor_cost_bwd[r,i,j] = Sol["XOR_Cost_bwd[%d,%d,%d]" %(r,i,j)]
    
    for r in range(total_round):
        for j in COL:
            mc_cost_fwd[r,j] = Sol["MC_Cost_fwd[%d,%d]" %(r,j)]
            mc_cost_bwd[r,j] = Sol["MC_Cost_bwd[%d,%d]" %(r,j)]

    for i in ROW:
        for j in COL:
            ini_enc_x[i,j] = Sol["E_ini_x[%d,%d]" %(i,j)]
            ini_enc_y[i,j] = Sol["E_ini_y[%d,%d]" %(i,j)]
            ini_enc_g[i,j] = Sol["E_ini_g[%d,%d]" %(i,j)]
    
    for i in ROW:
        for j in range(Nk):
            ini_key_x[i,j] = Sol["K_ini_x[%d,%d]" %(i,j)]
            ini_key_y[i,j] = Sol["K_ini_y[%d,%d]" %(i,j)]
            ini_key_g[i,j] = Sol["K_ini_g[%d,%d]" %(i,j)]

    for i in ROW:
        for j in COL:
            Meet_fwd_x[i,j] = Sol["Meet_fwd_x[%d,%d]" %(i,j)]
            Meet_fwd_y[i,j] = Sol["Meet_fwd_y[%d,%d]" %(i,j)]
            Meet_bwd_x[i,j] = Sol["Meet_bwd_x[%d,%d]" %(i,j)]
            Meet_bwd_y[i,j] = Sol["Meet_bwd_y[%d,%d]" %(i,j)]
    
    for j in COL:
        meet[j] = Sol["Meet[%d]" %j]
        meet_s[j] = Sol["Meet_signed[%d]" %j]

    ini_df_enc_b = np.sum(ini_enc_x[:,:]) - np.sum(ini_enc_g[:,:])
    ini_df_enc_r = np.sum(ini_enc_y[:,:]) - np.sum(ini_enc_g[:,:])

    ini_df_key_b = np.sum(ini_key_x[:,:]) - np.sum(ini_key_g[:,:])
    ini_df_key_r = np.sum(ini_key_y[:,:]) - np.sum(ini_key_g[:,:])

    DF_b = Sol["DF_b"]
    DF_r = Sol["DF_r"]
    Match = Sol["Match"]
    GnD_b = Sol["GND_b"]
    GnD_r = Sol["GND_r"]
    GnD_br = Sol["GND_br"]
    Obj = Sol["Obj"]



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

    
    

    outfile = './x'
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
    NROW =0
    NCOL = 0
    for r in range(total_round):
        mc_cost_fwd_col = 0
        mc_cost_bwd_col = 0
        for i in COL:
            mc_cost_fwd_col += mc_cost_fwd[r, i]
            mc_cost_bwd_col += mc_cost_fwd[r, i]
        
        indent = 0
        ## SB
        fid.write('\\begin{scope}[yshift =' + str(- r * (NROW + HO))+' cm, xshift =' + str(indent * (NCOL + WO))+' cm]'+'\n')
        for i in ROW:
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
        if r in bwd:
            fid.write('\\draw[edge, <-] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
        else:
            fid.write('\\draw[edge, ->] (' + str(NCOL) + ',' + str(NROW//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
        if r == enc_start_round:
            fid.write('\\path (' + str(NCOL//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_d1) + '~\\DoFF,~+' + str(ini_d2) + '~\\DoFB)$};'+'\n')
            fid.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        
        
        indent = indent + 1
        ## MC
        fid.write('\\begin{scope}[yshift =' + str(- r * (NROW + HO))+' cm, xshift =' +str(indent * (NCOL + WO))+' cm]'+'\n')
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
        if r == TR - 1 and WLastMC == 0:
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
            fid.write('\\path (' + str((NCOL + WO) - WO//2) + ',' + str(-0.8) + ') node {\\scriptsize$ (-' + str(mc_cost_fwd_col) + '~\\DoFF,~-' + str(mc_cost_bwd_col) + '~\\DoFB)$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')

        indent = indent + 1
        ## SB r+1
        fid.write('\\begin{scope}[yshift =' + str(- r * (NROW + HO))+' cm, xshift =' +str(indent * (NCOL + WO))+' cm]'+'\n')
        for i in range(NROW):
            row = NROW - 1 - i
            for j in range(NCOL):
                col = j
                fid.write(CM[SB_x_v[(r+1)%TR,i,j], SB_y_v[(r+1)%TR,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
        fid.write('\\draw (0,0) rectangle (' + str(NCOL) + ',' + str(NROW) + ');' + '\n')
        for i in range(1, NROW):
            fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCOL) + ',' + str(0) + ');' + '\n')
        for i in range(1, NCOL):
            fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NROW) + ');' + '\n')
        fid.write('\\path (' + str(NCOL//2) + ',' + str(NROW + 0.5) + ') node {\\scriptsize$\\SB^' + str((r+1)%TR) + '$};'+'\n')
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')
    ## Final
    fid.write('\\begin{scope}[yshift =' + str(- TR * (NROW + HO) + HO)+' cm, xshift =' +str(2 * (NCOL + WO))+' cm]'+'\n')
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



    #fid.write('\n'+'\\end{scope}'+'\n')
    fid.write('\n\n')
    fid.write('\\end{tikzpicture}'+'\n\n'+'\\end{document}')
    fid.close()
    from os import system
    system("pdflatex -output-directory='./' ./" + outfile + ".tex") 
    
    fid.close()

    return



    f =  open(dir + 'Vis_' + model_name + '_sol_' + str(sol_i) + '.txt', 'w')
    f.write('Model:\n')
    f.write(TAB+ 'Total: ' + str(total_round) +'\n')
    f.write(TAB+ 'Start at: r' + str(enc_start_round) +'\n')
    f.write(TAB+ 'Meet at: r' + str(match_round) +'\n')
    f.write(TAB+ 'KEY start at: r' + str(key_start_round) +'\n')
    f.write('\nInitialization:\n')
    f.write(TAB+'ENC FWD: ' + str(ini_df_enc_b) + '\n' + TAB+ 'ENC BWD: ' + str(ini_df_enc_r) + '\n')
    f.write(TAB+'KEY FWD: ' + str(ini_df_key_b) + '\n' + TAB+ 'KEY BWD: ' + str(ini_df_key_r) + '\n')
    f.write('\nSolution:\n'+TAB+'Obj := min{DF_b=%d - GnD_b=%d, DF_r=%d - GnD_r=%d, Match=%d - GnD_b - GnD_r - GnD_br=%d} = %d' %(DF_b, GnD_b, DF_r, GnD_r, Match, GnD_br, Obj) + '\n')
    f.write('\nVisualization:\n')
    
    for r in range(total_round):
        header = "r%d  " %r 
        
        if r == match_round:
            header+= 'mat -><-'
        elif r in fwd:
            header+= 'fwd --->'
        elif r in bwd:
            header+= 'bwd <---'
        if r == enc_start_round:
            header+=TAB*2 + 'ENC_start'
        
        f.write(header + '\n')
        nr = (r+1)%total_round
        
        line1 = ''
        line2 = ''
        for i in ROW:
            SB, MC, fMC, bMC, fAK, bAK, fKEY, bKEY, fSB, bSB, SBN, KEY = '','','','','','','','','','','','' 
            for j in COL:
                SB+=color(SB_x[r,i,j], SB_y[r,i,j])
                MC+=color(MC_x[r,i,j], MC_y[r,i,j])
                fMC+=color(fMC_x[r,i,j], fMC_y[r,i,j])
                bMC+=color(bMC_x[r,i,j], bMC_y[r,i,j])
                fAK+=color(fAK_x[r,i,j], fAK_y[r,i,j])
                bAK+=color(bAK_x[r,i,j], bAK_y[r,i,j])
                
                fKEY+=color(fKEY_x[r,i,j], fKEY_y[r,i,j])
                bKEY+=color(bKEY_x[r,i,j], bKEY_y[r,i,j])

                fSB+=color(fSB_x[nr,i,j], fSB_y[nr,i,j])
                bSB+=color(bSB_x[nr,i,j], bSB_y[nr,i,j])
                SBN+=color(SB_x[nr,i,j], SB_y[nr,i,j])
            
            if r == total_round - 1:
                fAK = '////'
                bAK = '////'

            line1 += SB + TAB*2 + MC + TAB*2+ fMC + TAB*2 + fAK + TAB*2 + fKEY + TAB*2 + fSB + TAB*2 + SBN + '\n'
            line2 += TAB+ TAB*2 + TAB+ TAB*2+ bMC + TAB*2 + bAK + TAB*2 + bKEY + TAB*2 + bSB + '\n' 
        
        f.write('SB#%d'%r +TAB*2+'MC#%d' %r +TAB*2+'fMC#%d   ' %r + TAB +'fAK#%d   '%r +TAB+'fKEY#%d  '%r +TAB+ 'fSB#%d   '%nr +TAB+ 'SB#%d'%nr+ '\n')
        f.write(line1 + '\n')
        f.write(TAB*3           +TAB*3            +'bMC#%d   ' %r + TAB +'bAK#%d   '%r +TAB+'bKEY#%d  '%r +TAB+ 'bSB#%d   '%nr + '\n')
        f.write(line2 + '\n')
        
        if mc_cost_fwd[r,:].any() or mc_cost_bwd[r,:].any():
            f.write('MixCol costs fwdDf: '+ str(mc_cost_fwd[r,:]) + TAB+ 'bwdDf: ' +str(mc_cost_bwd[r,:])+ '\n')
        if xor_cost_fwd[r,:,:].any():
            f.write('AddKey costs fwdDf: ' + '\n' + str(xor_cost_fwd[r,:,:]) + '\n')
        if xor_cost_bwd[r,:,:].any():
                f.write('AddKey costs bwdDf: ' + '\n' + str(xor_cost_bwd[r,:,:]) + '\n')
        f.write('\n')
        
        if r == match_round and match_round != total_round - 1:
            f.write('Match Thru MC:'+'\n'+ 'MC#%d' %r +TAB*2+ 'Meet_BWD' +'\n')
            for i in ROW:
                MC = ''
                Meet_B = ''
                for j in COL:
                    MC+=color(MC_x[r,i,j], MC_y[r,i,j])
                    Meet_B+=color(Meet_bwd_x[i,j], Meet_bwd_y[i,j])
                f.write(MC+TAB*2+Meet_B+'\n') 
            f.write('Degree of Matching:' + str(meet[:]) + '\n'*2)

    # process whiten key
    r = -1
    nr = 0
    line1 = ''
    line2 = ''
    for i in ROW:
        fKEY, bKEY, fAT, bAT, fNSB, bNSB, NSB = '', '', '', '', '', '', ''
        for j in COL:
            fAT +=color(fAK_x[r,i,j], fAK_y[r,i,j])
            bAT +=color(bAK_x[r,i,j], bAK_y[r,i,j])
            fKEY +=color(fKEY_x[r,i,j], fKEY_y[r,i,j])
            bKEY +=color(bKEY_x[r,i,j], bKEY_y[r,i,j])
            fNSB += color(fSB_x[0,i,j], fSB_y[0,i,j])
            bNSB += color(bSB_x[0,i,j], bSB_y[0,i,j])
            NSB += color(SB_x[nr,i,j], SB_y[nr,i,j])
        if match_round == total_round -1:
                NSB = '////'
        line1 += 9*TAB + fAT+ TAB*2 + fKEY + '\n'
        line2 += 9*TAB + bAT+ TAB*2 + bKEY + '\n'
    f.write(TAB*9 +'fAT     ' +TAB+'fKEY#%d  '%r +TAB+ 'fSB#%d   '%nr +TAB+ 'SB#%d' %nr + '\n')
    f.write(line1 + '\n')
    f.write(TAB*9 +'bAT     ' +TAB+'bKEY#%d  '%r +TAB+ 'bSB#%d   '%nr + '\n')
    f.write(line2 + '\n')
    
    tr = total_round
    if xor_cost_fwd[tr,:,:].any():
            f.write('AddKey costs fwdDf: ' + '\n' + str(xor_cost_fwd[tr,:,:]) + '\n')
    if xor_cost_bwd[tr,:,:].any():
            f.write('AddKey costs bwdDf: ' + '\n' + str(xor_cost_bwd[tr,:,:]) + '\n')
    
    if match_round == total_round - 1:
        f.write('\n' + "Identity Match:" + '\n')
        f.write('Meet_FWD'+ TAB + 'SB#0' + '\n')
        for i in ROW:
            SB = ''
            MF = ''
            for j in COL:
                SB +=color(SB_x[0,i,j], SB_y[0,i,j])
                MF +=color(Meet_fwd_x[i,j], Meet_fwd_y[i,j])
            f.write(MF+ TAB*2 + SB + '\n')

    f.write('\n'+'Key Schedule: starts at r'+str(key_start_round)+'\n')
    
    for r in range(Nr):
        f.write('KEY_SCHEDULE_'+str(r)+'\n')
        line1 = ''
        line2 = ''
        for i in ROW:
            for j in range(Nk):
                line1 += color(fKSch_x[r,i,j], fKSch_y[r,i,j])
                line2 += color(bKSch_x[r,i,j], bKSch_y[r,i,j])
                if j == Nk-1 and r!= 0 and r!=Nr-1:
                    line1 += TAB + color(KSub_x[r,(i-1)%4], KSub_y[r,(i-1)%4]) + TAB + color(KSub_x[r,i], KSub_y[r,i]) + TAB + color(fKSub_x[r,i], fKSub_y[r,i])
                    line2 += TAB + ' '                                         + TAB + ' '                             + TAB + color(bKSub_x[r,i], bKSub_y[r,i])
            line1+='\n'
            line2+='\n'
        f.write(line1+'\n'+line2)
        f.write('\n'*2)

        if Key_cost_fwd[r,:,:].any():
            f.write('KeyExp costs fwdDf: ' + '\n' + str(Key_cost_fwd[r,:,:]) + '\n')
        if Key_cost_bwd[r,:,:].any():
            f.write('KeyExp costs bwdDf: ' + '\n' + str(Key_cost_bwd[r,:,:]) + '\n')
        f.write('\n'*2)   

    f.close()



displaySol(192, 9, 3, 8, 3, 'AES%dRK_%dr_ENC_r%d_Meet_r%d_KEY_r%d' % (192,9,3,8,3), 0, dir='./AES_SupP_GnD_RKcorrection/runs/')