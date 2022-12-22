# simple txt display of the solution
from io import TextIOWrapper
import gurobipy as gp
from gurobipy import GRB
from string import Template
import numpy as np
import re
import os
import math
import copy

# AES parameters
NROW = 4
NCOL = 4
NBYTE = 32
NGRID = NROW * NCOL
NBRANCH = NROW + 1
ROW = range(NROW)
COL = range(NCOL)
TAB = ' ' * 4


def tex_display(key_size:int, total_round:int, enc_start_round:int, match_round:int, key_start_round:int, model_name:str, sol_i:int, time:int, dir:str):
    def draw_gridlines(file, id = 'ENC'):
        if id == 'ENC':
            ncol = NCOL
        elif id == 'KS': 
            ncol = Nk
        else: 
            ncol = 1
        
        file.write('%' +' draw grid lines:\n')
        file.write('\\draw (0,0) rectangle (%d,%d);\n'   %(ncol, NROW))
        for i in range(1, NROW):
            file.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(ncol) + ',' + str(0) + ');' + '\n')
        for i in range(1, ncol):
            file.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NROW) + ');' + '\n')

    def draw_cells(W_x, W_y, file):
        if W_x.shape != W_y.shape:
            return
        if len(W_x.shape) > 2:  # chained states
            #return
            nrow = len(W_x[0])
            ncol = len(W_x[0][0])
            for i in range(nrow):
                row = nrow - 1 - i
                for j in range(ncol):
                    col = j
                    file.write(color_fill[W_x[r,i,j], W_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(col, row))
        else:   # single state
            if W_x.shape == (NROW,NCOL):
                nrow = 4
                ncol = 4
                for ri in range(nrow):
                    i = nrow - 1 - ri
                    for j in range(ncol):
                        file.write(color_fill[W_x[i,j], W_y[i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, i))
            elif W_x.shape == (NROW,1):
                nrow = 4
                ncol = 1
                for ri in range(nrow):
                    i = nrow - 1 - ri
                    #for j in range(ncol):
                    file.write(color_fill[W_x[r,i], W_y[r,i]] + ' (%d,%d) rectangle + (1,1);\n'   %(0, i))
            elif W_x.shape == (Nr,NROW):
                nrow = 4
                for ri in range(nrow):
                    i = nrow - 1 - ri
                    #for j in range(ncol):
                    file.write(color_fill[W_x[r,i], W_y[r,i]] + ' (%d,%d) rectangle + (1,1);\n'   %(0, i))

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

#### Register variables  
    S_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    S_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fS_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fS_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bS_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bS_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    M_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    M_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fM_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fM_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bM_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bM_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fAR_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fAR_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAR_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAR_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fAL_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    fAL_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAL_x = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    bAL_y = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    
    fK_x = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fK_y = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bK_x = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bK_y = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

    fK_inv_x = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fK_inv_y = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bK_inv_x = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bK_inv_y = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

    fKL_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fKL_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKL_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKL_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

    fKR_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    fKR_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKR_x= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    bKR_y= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    

    fKS_x= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    fKS_y= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKS_x= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    bKS_y= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)

    KSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    KSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    fKSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    fKSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    bKSub_x= np.ndarray(shape=(Nr, NROW), dtype=int)
    bKSub_y= np.ndarray(shape=(Nr, NROW), dtype=int)
    
    ind_MulAK = np.ndarray(shape=(total_round+1, NCOL), dtype=int)
    
    Key_cost_fwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    Key_cost_bwd= np.ndarray(shape=(Nr, NROW, Nk), dtype=int)
    xor_rhs_cost_fwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    xor_rhs_cost_bwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    xor_lhs_cost_fwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    xor_lhs_cost_bwd = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
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

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                S_x[r,i,j]=Sol["S_x[%d,%d,%d]" %(r,i,j)]
                S_y[r,i,j]=Sol["S_y[%d,%d,%d]" %(r,i,j)]

                M_x[r,i,j]=Sol["M_x[%d,%d,%d]" %(r,i,j)]
                M_y[r,i,j]=Sol["M_y[%d,%d,%d]" %(r,i,j)]

                fS_x[r,i,j]=Sol["fS_x[%d,%d,%d]" %(r,i,j)]
                fS_y[r,i,j]=Sol["fS_y[%d,%d,%d]" %(r,i,j)]
                bS_x[r,i,j]=Sol["bS_x[%d,%d,%d]" %(r,i,j)]
                bS_y[r,i,j]=Sol["bS_y[%d,%d,%d]" %(r,i,j)]
                

                fAR_x[r,i,j]=Sol["fAR_x[%d,%d,%d]" %(r,i,j)]
                fAR_y[r,i,j]=Sol["fAR_y[%d,%d,%d]" %(r,i,j)]
                bAR_x[r,i,j]=Sol["bAR_x[%d,%d,%d]" %(r,i,j)]
                bAR_y[r,i,j]=Sol["bAR_y[%d,%d,%d]" %(r,i,j)]

                fAL_x[r,i,j]=Sol["fAL_x[%d,%d,%d]" %(r,i,j)]
                fAL_y[r,i,j]=Sol["fAL_y[%d,%d,%d]" %(r,i,j)]
                bAL_x[r,i,j]=Sol["bAL_x[%d,%d,%d]" %(r,i,j)]
                bAL_y[r,i,j]=Sol["bAL_y[%d,%d,%d]" %(r,i,j)]

                fM_x[r,i,j]=Sol["fM_x[%d,%d,%d]" %(r,i,j)]
                fM_y[r,i,j]=Sol["fM_y[%d,%d,%d]" %(r,i,j)]
                bM_x[r,i,j]=Sol["bM_x[%d,%d,%d]" %(r,i,j)]
                bM_y[r,i,j]=Sol["bM_y[%d,%d,%d]" %(r,i,j)]

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                continue
                M_x[r,i,j] = S_x[r,i,(j+i)%NCOL]
                M_y[r,i,j] = S_y[r,i,(j+i)%NCOL]
    
    KeyS_r = 0
    KeyS_j = 0
    for r in range(-1, total_round):
        for j in COL:
            for i in ROW:
                fK_x[r,i,j] = Sol["fKS_x[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                fK_y[r,i,j] = Sol["fKS_y[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bK_x[r,i,j] = Sol["bKS_x[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
                bK_y[r,i,j] = Sol["bKS_y[%d,%d,%d]" %(KeyS_r,i,KeyS_j)]
            
            KeyS_j += 1
            if KeyS_j % Nk == 0:
                KeyS_r += 1
                KeyS_j = 0
    
    for r in range(total_round+1):
        for i in ROW:
            for j in COL:
                fK_inv_x[r,i,j] = Sol["fK_invMC_x[%d,%d,%d]" %(r,i,j)]
                fK_inv_y[r,i,j] = Sol["fK_invMC_y[%d,%d,%d]" %(r,i,j)]
                bK_inv_x[r,i,j] = Sol["bK_invMC_x[%d,%d,%d]" %(r,i,j)]
                bK_inv_y[r,i,j] = Sol["bK_invMC_y[%d,%d,%d]" %(r,i,j)]
    
    for r in range(Nr):
        for i in ROW:
            for j in range(Nk):
                fKS_x[r,i,j] = Sol["fKS_x[%d,%d,%d]" %(r,i,j)]
                fKS_y[r,i,j] = Sol["fKS_y[%d,%d,%d]" %(r,i,j)]
                bKS_x[r,i,j] = Sol["bKS_x[%d,%d,%d]" %(r,i,j)]
                bKS_y[r,i,j] = Sol["bKS_y[%d,%d,%d]" %(r,i,j)]

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
                fKL_x[r,i,j] = Sol["fKL_x[%d,%d,%d]" %(r,i,j)]
                fKL_y[r,i,j] = Sol["fKL_y[%d,%d,%d]" %(r,i,j)]
                bKL_x[r,i,j] = Sol["bKL_x[%d,%d,%d]" %(r,i,j)]
                bKL_y[r,i,j] = Sol["bKL_y[%d,%d,%d]" %(r,i,j)]

                fKR_x[r,i,j] = Sol["fKR_x[%d,%d,%d]" %(r,i,j)]
                fKR_y[r,i,j] = Sol["fKR_y[%d,%d,%d]" %(r,i,j)]
                bKR_x[r,i,j] = Sol["bKR_x[%d,%d,%d]" %(r,i,j)]
                bKR_y[r,i,j] = Sol["bKR_y[%d,%d,%d]" %(r,i,j)]


    for r in range(total_round+1):
        for i in ROW:
            for j in COL:
                xor_rhs_cost_fwd[r,i,j] = Sol["XOR_RHS_Cost_fwd[%d,%d,%d]" %(r,i,j)]
                xor_rhs_cost_bwd[r,i,j] = Sol["XOR_RHS_Cost_bwd[%d,%d,%d]" %(r,i,j)]
                xor_lhs_cost_fwd[r,i,j] = Sol["XOR_LHS_Cost_fwd[%d,%d,%d]" %(r,i,j)]
                xor_lhs_cost_bwd[r,i,j] = Sol["XOR_LHS_Cost_bwd[%d,%d,%d]" %(r,i,j)]
    
    for r in range(total_round+1):
        for j in COL:
            ind_MulAK[r,j] = Sol["MulAK_indicator[%d,%d]" %(r,j)]
    
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
            Meet_fwd_x[i,j] = Sol["Meet_lhs_x[%d,%d]" %(i,j)]
            Meet_fwd_y[i,j] = Sol["Meet_lhs_y[%d,%d]" %(i,j)]
            Meet_bwd_x[i,j] = Sol["Meet_rhs_x[%d,%d]" %(i,j)]
            Meet_bwd_y[i,j] = Sol["Meet_rhs_y[%d,%d]" %(i,j)]
    
    for j in COL:
        meet[j] = Sol["Meet[%d]" %j]

    ini_df_enc_b = np.sum(ini_enc_x[:,:]) - np.sum(ini_enc_g[:,:])
    ini_df_enc_r = np.sum(ini_enc_y[:,:]) - np.sum(ini_enc_g[:,:])

    ini_df_key_b = np.sum(ini_key_x[:,:]) - np.sum(ini_key_g[:,:])
    ini_df_key_r = np.sum(ini_key_y[:,:]) - np.sum(ini_key_g[:,:])

    start_df_b = ini_df_enc_b + ini_df_key_b
    start_df_r = ini_df_enc_r + ini_df_key_r

    DF_b = Sol["DF_b"]
    DF_r = Sol["DF_r"]
    Match = Sol["Match"]
    GnD_b = Sol["GND_b"]
    GnD_r = Sol["GND_r"]
    GnD_br = Sol["GND_br"]
    Obj = Sol["Obj"]

    color_fill = np.ndarray(shape=(2, 2),dtype='object')
    color_fill[0, 0] = '\\fill[\\UW]'
    color_fill[0, 1] = '\\fill[\\BW]'
    color_fill[1, 0] = '\\fill[\\FW]'
    color_fill[1, 1] = '\\fill[\\CW]'
    
    if NROW == 4:
        # shift notes the space that the message is taking
        y_shift = NROW*2
        x_shift = NCOL
        # tab notes the indent
        xtab = x_shift + NCOL
        ytab = y_shift + NROW*3

    pdfname = 'AES%d_%d%d%d%d_obj%d_sol%d' % (key_size, total_round, enc_start_round, match_round, key_start_round, Obj, sol_i)
    f = open(dir + pdfname + '.tex', 'w')
    
    # write latex header
    f.write( '%' + ' Vis_' + model_name + '\n'
        '\\documentclass{standalone}' + '\n'
        '\\usepackage[usenames,dvipsnames]{xcolor}' + '\n'
        '\\usepackage{amsmath,amssymb,mathtools,tikz,calc,pgffor,import}' + '\n'
        '\\usepackage{xspace}' + '\n'
        '\\usetikzlibrary{crypto.symbols,patterns,calc}' + '\n'
        '\\tikzset{shadows=no}' + '\n'
        '\\input{macro}' + '\n')
    
    f.write( '%' + 'document starts' + '\n'
        '\\begin{document}' + '\n' +
        '\\begin{tikzpicture}[scale=0.2, every node/.style={font=\\boldmath\\bf}]' + '\n'
	    '\\everymath{\\scriptstyle}' + '\n'
	    '\\tikzset{edge/.style=->, >=stealth, arrow head=8pt, thick};' + '\n')
    
    # borderline
    f.write('%'+'borderline\n' + '\\draw -- (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d);\n\n'
    %(
        -1*xtab, ytab,
        12*(Nr+x_shift), ytab,
        12*(Nr+x_shift), -(total_round+2)*ytab,
        -1*xtab, -(total_round+2)*ytab
    ))
    
    # draw enc states
    for r in range(total_round - 1):
        
        if r == match_round:
            arrow = '->'
            op1, op2 = 'SupP', 'SupP'
        elif r in fwd:
            arrow = '->'
            op1, op2 = 'SupP', 'Eval'
        else:
            arrow = '<-'
            op1, op2 = 'Eval', 'SupP'
        
        # SB
        slot = 0
        f.write('%' + ' Round %d SubByte\n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[S_x[r,i,j], S_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, r))
        if r == enc_start_round:
            f.write('\\path (' + str(NCOL//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_df_enc_b) + '~\\DoFF,~+' + str(ini_df_enc_r) + '~\\DoFB)$};'+'\n')
            f.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
        f.write('\\end{scope}\n')
        
        # SB&SR link
        slot = 0
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(%d,0);\n' %(arrow, 0, NROW//2, NCOL))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')

        # MC
        slot += 1
        f.write('%' + ' Round %d MixCol \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[M_x[r,i,j], M_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%d$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')
        
        # SupP link
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny %s} + (%d,0);\n'    %(arrow, 0, NROW//2, op1, NCOL))
        f.write('\\filldraw [black] (%d,%d) circle (4 pt);\n'    %(NCOL//2, NROW//2))
        if op1 == 'SupP':
            f.write('\\draw[edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(arrow, NCOL//2, NROW//2, NCOL//2, -NROW, NCOL, -NROW))
        else: 
            f.write('\\draw (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(NCOL//2, NROW//2, NCOL//2, -NROW, NCOL, -NROW))
        f.write('\\end{scope}\n')

        # fM and bM
        slot += 1
        f.write('%' + ' Round %d FWD MixCol \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fM_x[r,i,j], fM_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD MixCol \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bM_x[r,i,j], bM_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        # add KL link
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AKL} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
        f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
        f.write('\\end{scope}\n')

        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-1.5*NROW))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AKL} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
        f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
        f.write('\\end{scope}\n')

        # Add eq LHS key cost
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-2*NROW))
        if xor_lhs_cost_fwd[r,:].any() or xor_lhs_cost_bwd[r,:].any():
            f.write('\\path (%f,%f) node {\\tiny Add KL};\n'  %(NCOL//2, -NROW//4))
            f.write('\\path (%f,%f) node {\\scriptsize$(-%d ~\\DoFF, -%d ~\\DoFB)$};\n'  %(NCOL//2, -NROW//2, np.sum(xor_lhs_cost_fwd[r,:]), np.sum(xor_lhs_cost_bwd[r,:])))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')

        # fAL and bAL
        slot += 1
        f.write('%' + ' Round %d FWD AL \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fAL_x[r,i,j], fAL_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\AL^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD AL \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bAL_x[r,i,j], bAL_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\AL^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        # add MC link
        if r == match_round and r != total_round - 1:
            f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
            f.write('\\draw (%f,%f) node[below] {\\tiny \\scriptsize$\\StMeet$};\n' %(1.5*NCOL,0))
            f.write('\\draw (%f,%f) node[above] {\\tiny \\scriptsize$\\EndFwd$};\n' %(0.5*NCOL,-2*NROW))
            f.write('\\draw (%f,%f) node[above] {\\tiny \\scriptsize$\\EndBwd$};\n' %(2.5*NCOL,-2*NROW))
            f.write('\\end{scope}\n')

            if arrow == '->':
                arrow = '<-'
            else: 
                arrow = '->'
        else: 
            f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
            f.write('\\end{scope}\n')

            f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-1.5*NROW))
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
            f.write('\\end{scope}\n')
    
        # MC cost
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-2*NROW))
        if mc_cost_fwd[r,:].any() or mc_cost_bwd[r,:].any():
            f.write('\\path (%f,%f) node {\\tiny MC};\n'  %(NCOL//2, -NROW//4))
            f.write('\\path (%f,%f) node {\\scriptsize$(-%d ~\\DoFF, -%d ~\\DoFB)$};\n'  %(NCOL//2, -NROW//2, np.sum(mc_cost_fwd[r,:]), np.sum(mc_cost_bwd[r,:])))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')


        # fAR and bAR
        slot += 1
        f.write('%' + ' Round %d FWD AR \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fAR_x[r,i,j], fAR_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\AR^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD AR \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bAR_x[r,i,j], bAR_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\AR^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        # add AKR link
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AKR} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
        f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
        f.write('\\end{scope}\n')

        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-1.5*NROW))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AKR} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
        f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
        f.write('\\end{scope}\n')
            
        # SupP SubByte state (next round)
        slot += 1
        f.write('%' + ' Round %d FWD NSB \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fS_x[(r+1)%total_round,i,j], fS_y[(r+1)%total_round,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dF$};\n'    %(NCOL//2, NROW+0.5, (r+1)%total_round))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD NSB \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bS_x[(r+1)%total_round,i,j], bS_y[(r+1)%total_round,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dB$};\n'    %(NCOL//2, NROW+0.5, (r+1)%total_round))
        f.write('\\end{scope}\n')


        # NSB
        slot += 1
        f.write('%' + ' Round %d NSB \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[S_x[(r+1)%total_round,i,j], S_y[(r+1)%total_round,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, (r+1)%total_round))
        f.write('\\end{scope}\n')
        
        # SupP link
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab-NCOL, -r*ytab))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny %s} + (%d,0);\n'    %(arrow, 0, NROW//2, op2, NCOL))
        f.write('\\filldraw [black] (%d,%d) circle (4 pt);\n'    %(NCOL//2, NROW//2))
        if op2 == 'SupP':
            f.write('\\draw[edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(arrow, 0, -NROW, NCOL//2, -NROW, NCOL//2, NROW//2))
        else: 
            f.write('\\draw (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(NCOL//2, NROW//2, NCOL//2, -NROW, 0, -NROW))
        f.write('\\end{scope}\n')

        # enc state link
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        f.write('\\draw [edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d);\n'    
        %( arrow,
        NCOL//2, 0, 
        NCOL//2, -3*NROW, 
        -NCOL-slot*xtab, -3*NROW, 
        -NCOL-slot*xtab, -4.5*NROW, 
        -slot*xtab, -4.5*NROW))
        f.write('\\end{scope}\n')

        # fKL and bKL
        slot += 1
        f.write('%' + ' Round %d FWD KL \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fKL_x[r,i,j], fKL_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\KL^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD KL \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bKL_x[r,i,j], bKL_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\KL^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        # fKR and bKR
        slot += 1
        f.write('%' + ' Round %d FWD KR \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fKR_x[r,i,j], fKR_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\KR^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD KR \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bKR_x[r,i,j], bKR_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\KR^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')
        
        # fK and bK
        slot += 1
        f.write('%' + ' Round %d FWD K \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fK_x[r,i,j], fK_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD K \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bK_x[r,i,j], bK_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        # add MulAK inv MC link
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC} + (%d,0) ;\n'    %('<-', 0, NROW//2, NCOL))
        f.write('\\end{scope}\n')

        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-1.5*NROW))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC} + (%d,0) ;\n'    %('<-', 0, NROW//2, NCOL))
        f.write('\\end{scope}\n')

        # fKinv and bKinv
        slot += 1
        f.write('%' + ' Round %d FWD Kinv \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in ROW:
            for j in COL:
                f.write(color_fill[fK_inv_x[r,i,j], fK_inv_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\Kinv^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('%' + ' Round %d BWD Kinv \n' %r)
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
        for i in ROW:
            for j in COL:
                f.write(color_fill[bK_inv_x[r,i,j], bK_inv_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\Kinv^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')
        
    


    # last round
    r = total_round - 1
    # SB
    slot = 0
    f.write('%' + ' Round %d SubByte\n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[S_x[r,i,j], S_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, r))
    if r == enc_start_round:
        f.write('\\path (' + str(NCOL//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_df_enc_b) + '~\\DoFF,~+' + str(ini_df_enc_r) + '~\\DoFB)$};'+'\n')
        f.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
    f.write('\\end{scope}\n')
    
    # SB&SR link
    slot = 0
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(%d,0);\n' %(arrow, 0, NROW//2, NCOL))
    f.write('\n'+'\\end{scope}'+'\n')
    f.write('\n\n')

    # MC
    slot += 1
    f.write('%' + ' Round %d MixCol \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[M_x[r,i,j], M_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%d$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')
    
    # SupP link
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny %s} + (%d,0);\n'    %(arrow, 0, NROW//2, op1, NCOL))
    f.write('\\filldraw [black] (%d,%d) circle (4 pt);\n'    %(NCOL//2, NROW//2))
    if op1 == 'SupP':
        f.write('\\draw[edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(arrow, NCOL//2, NROW//2, NCOL//2, -NROW, NCOL, -NROW))
    else: 
        f.write('\\draw (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(NCOL//2, NROW//2, NCOL//2, -NROW, NCOL, -NROW))
    f.write('\\end{scope}\n')

    # fM and bM
    slot += 1
    f.write('%' + ' Round %d FWD MixCol \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[fM_x[r,i,j], fM_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    f.write('%' + ' Round %d BWD MixCol \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
    for i in ROW:
        for j in COL:
            f.write(color_fill[bM_x[r,i,j], bM_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    # add KL link
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AK} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
    f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
    f.write('\\end{scope}\n')

    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-1.5*NROW))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AK} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
    f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
    f.write('\\end{scope}\n')

    # Add key cost
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-2*NROW))
    if xor_rhs_cost_fwd[r,:].any() or xor_rhs_cost_bwd[r,:].any():
        f.write('\\path (%f,%f) node {\\tiny AK};\n'  %(NCOL//2, -NROW//4))
        f.write('\\path (%f,%f) node {\\scriptsize$(-%d ~\\DoFF, -%d ~\\DoFB)$};\n'  %(NCOL//2, -NROW//2, np.sum(xor_rhs_cost_fwd[r,:]), np.sum(xor_rhs_cost_bwd[r,:])))
    f.write('\n'+'\\end{scope}'+'\n')
    f.write('\n\n')
    
    # fAK and bAK
    slot += 1
    f.write('%' + ' Round %d FWD AK \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[fAR_x[r,i,j], fAR_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\AT^F$};\n'    %(NCOL//2, NROW+0.5))
    f.write('\\end{scope}\n')

    f.write('%' + ' Round %d BWD AK \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
    for i in ROW:
        for j in COL:
            f.write(color_fill[bAR_x[r,i,j], bAR_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\AT^B$};\n'    %(NCOL//2, NROW+0.5))
    f.write('\\end{scope}\n')

    # add AK link
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AK} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
    f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
    f.write('\\end{scope}\n')

    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-1.5*NROW))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny AK} + (%d,0) ;\n'    %(arrow, 0, NROW//2, NCOL))
    f.write('\\node[scale = %f, XOR] at (%f,%f){};'   %(0.5, NCOL//2, NROW//2))
    f.write('\\end{scope}\n')

    # Add key cost
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL, -r*ytab-2*NROW))
    if xor_rhs_cost_fwd[r+1,:].any() or xor_rhs_cost_bwd[r+1,:].any():
        f.write('\\path (%f,%f) node {\\tiny Add KL};\n'  %(NCOL//2, -NROW//4))
        f.write('\\path (%f,%f) node {\\scriptsize$(-%d ~\\DoFF, -%d ~\\DoFB)$};\n'  %(NCOL//2, -NROW//2, np.sum(xor_rhs_cost_fwd[r,:]), np.sum(xor_rhs_cost_bwd[r,:])))
    f.write('\\end{scope}\n')

    slot += 1
    f.write('%' + ' Round %d FWD SB \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[fS_x[0,i,j], fS_y[0,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    f.write('%' + ' Round %d BWD SB \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
    for i in ROW:
        for j in COL:
            f.write(color_fill[bS_x[0,i,j], bS_y[0,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    # NSB
    slot += 1
    f.write('%' + ' Round %d NSB \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[S_x[(r+1)%total_round,i,j], S_y[(r+1)%total_round,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, (r+1)%total_round))
    f.write('\\end{scope}\n')
    
    # SupP link
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab-NCOL, -r*ytab))
    f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny %s} + (%d,0);\n'    %(arrow, 0, NROW//2, op2, NCOL))
    f.write('\\filldraw [black] (%d,%d) circle (4 pt);\n'    %(NCOL//2, NROW//2))
    if op2 == 'SupP':
        f.write('\\draw[edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(arrow, 0, -NROW, NCOL//2, -NROW, NCOL//2, NROW//2))
    else: 
        f.write('\\draw (%d,%d) -- (%d,%d) -- (%d,%d);\n'   %(NCOL//2, NROW//2, NCOL//2, -NROW, 0, -NROW))
    f.write('\\end{scope}\n')

    # enc state link
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    f.write('\\draw [edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d);\n' 
    %( arrow,
    NCOL//2, 0, 
    NCOL//2, -3*NROW, 
    NCOL//2-(slot+1)*xtab, -3*NROW,
    NCOL//2-(slot+1)*xtab, r*ytab+NROW//2, 
    -slot*xtab, r*ytab+NROW//2))
    f.write('\\end{scope}\n')

    slot += 2
    f.write('%' + ' Round %d FWD K \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[fK_x[r,i,j], fK_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    f.write('%' + ' Round %d BWD K \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
    for i in ROW:
        for j in COL:
            f.write(color_fill[bK_x[r,i,j], bK_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    slot += 1
    f.write('%' + ' Round %d FWD K \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
    for i in ROW:
        for j in COL:
            f.write(color_fill[fK_x[total_round,i,j], fK_y[total_round,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')

    f.write('%' + ' Round %d BWD K \n' %r)
    f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-1.5*NROW))
    for i in ROW:
        for j in COL:
            f.write(color_fill[bK_x[total_round,i,j], bK_y[total_round,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
    draw_gridlines(f)
    f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
    f.write('\\end{scope}\n')





    slot = 12      
    # draw key schedule
    for r in range(Nr):
        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab))
        for i in range(Nb):
            for j in range(Nk):
                f.write(color_fill[fKS_x[r,i,j], fKS_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f, 'KS')
        f.write('\\path (%f,%f) node {\\scriptsize$\\KE^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab, -r*ytab-2*NROW))
        for i in range(Nb):
            for j in range(Nk):
                f.write(color_fill[bKS_x[r,i,j], bKS_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, NROW-1-i))
        draw_gridlines(f, 'KS')
        f.write('\\path (%f,%f) node {\\scriptsize$\\KE^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\end{scope}\n')

        f.write('\\begin{scope}[xshift = %f cm, yshift = %f cm]\n\n'   %(slot*xtab+NCOL//2, -r*ytab-2*NROW))
        if Key_cost_fwd[r,:].any() or Key_cost_bwd[r,:].any():
            f.write('\\path (%f,%f) node {\\tiny KeyExp};\n'  %(0, -NROW//4))
            f.write('\\path (%f,%f) node {\\scriptsize$(-%d ~\\DoFF, -%d ~\\DoFB)$};\n'  %(NCOL//2, -NROW//2, np.sum(Key_cost_fwd[r,:]), np.sum(Key_cost_bwd[r,:])))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')



        continue

        # Ksub
        slot += 1
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(Nk+x_shift) )) 
        draw_cells(KSub_x, KSub_y, f)
        draw_gridlines(f, 'KSub')
        f.write('\\path (%f,%f) node {\\scriptsize$\\temp^%d$};\n'    %(NCOL//2, NROW+0.5, r))
        #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny KeySchedule}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')

        # SupP Ksub
        slot += 0.75
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(Nk+x_shift) )) 
        draw_cells(fKSub_x, fKSub_y, f)
        draw_gridlines(f, 'KSub')
        f.write('\\path (%f,%f) node {\\scriptsize$\\temp^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny KeySchedule}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
        
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(Nk+x_shift) )) 
        draw_cells(bKSub_x, bKSub_y, f)
        draw_gridlines(f, 'KSub')
        f.write('\\path (%f,%f) node {\\scriptsize$\\temp^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny KeySschedule}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
    
    
    
    # different match settings
    if 1 == 0:
        # Match state
        # lengends
        f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round)*(ytab), (xtab))) 
        f.write('\\path (' + str(-NROW) + ',' + str(1.5) + ') node {\\scriptsize$\\StMatch$};'+'\n')
        f.write('\\draw[edge, ->] (%f,%f) -- +(%d,0);\n'    %(NCOL, NROW//2, x_shift//2))
        f.write('\\draw[edge, <-] (%f,%f) -- +(%d,0);\n'    %(NCOL+x_shift//2, NROW//2, x_shift//2))
        f.write('\\path (' + str(1.5*NROW) + ',' + str(3) + ') node {\\tiny Meet};'+'\n')
        f.write('\n'+'\\end{scope}'+'\n')
        if match_round == total_round - 1:
            # Meet_fwd
            f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round)*(ytab), (xtab))) 
            f.write('\\path (' + str(-NROW) + ',' + str(2.5) + ') node {\\tiny id};'+'\n')
            draw_cells(Meet_fwd_x, Meet_fwd_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\meet^F$};\n'    %(NCOL//2, NROW+0.5))
            f.write('\n'+'\\end{scope}'+'\n')
            # SB0
            r = 0
            f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round)*(ytab), 2*(xtab))) 
            draw_cells(S_x, S_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^0$};\n'    %(NCOL//2, NROW+0.5))
            f.write('\n'+'\\end{scope}'+'\n')
        else: 
            # MC
            r = match_round
            f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round)*(ytab), (xtab))) 
            draw_cells(M_x, M_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%d$};\n'    %(NCOL//2, NROW+0.5, r))
            f.write('\n'+'\\end{scope}'+'\n')
            # Meet_bwd
            f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round)*(ytab), 2*(xtab))) 
            draw_cells(Meet_bwd_x, Meet_bwd_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\meet^B$};\n'    %(NCOL//2, NROW+0.5))
            f.write('\n'+'\\end{scope}'+'\n')
    
    # display time cost
    f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round)*(ytab)-2*NROW, x_shift)) 
    f.write('\\node[draw] at (0,0) {time cost = %d sec};' %time)
    f.write('\n'+'\\end{scope}'+'\n')
    
    ## Final footnote
    f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round+1)*(3*NROW+y_shift), 2*(NCOL+x_shift))) 
    #f.write('\\begin{scope}[yshift =' + str(- total_round * (NROW + y_shift) + y_shift)+' cm, xshift =' +str(2 * (NCOL + x_shift))+' cm]'+'\n')
    f.write(
        '\\node[draw, thick, rectangle, text width=15cm, label={[shift={(-7,0)}]\\footnotesize Config}] at (16, 0) {\n' + 
	    '{\\footnotesize\n' + 
	    '$\\bullet~(\\varInitBL,~\\varInitRD)~=~(~+%d~\\DoFF, ~+%d~\\DoFB) ~~~ \\StENC=(~+%d~\\DoFF, ~+%d~\\DoFB) ~~~ \\StKSA=(~+%d~\\DoFF, ~+%d~\\DoFB)~$\n' %(start_df_b, start_df_r, ini_df_enc_b, ini_df_key_b, ini_df_enc_r, ini_df_key_r) +
        '\\\ \n' +
	    '$\\bullet~(\\varDoFBL,~\\varDoFRD,~\\varDoM, ~\\varDoFGESSBL, ~\\varDoFGESSRD, ~\\varDoFGESSBR)~=~(~+%d~\DoFF,~+%d~\DoFB,~+%d~\DoM,~-%d~\DoFFG,~-%d~\DoFB,~+%d~\DoFFBG))$\n' %(DF_b, DF_r, Match, GnD_b, GnD_r, GnD_br) +
        '}};'
        )
    f.write('\n'+'\\end{scope}'+'\n')
    
    f.write('\n\n')
    f.write('\\end{tikzpicture}'+'\n\n'+'\\end{document}')
    f.close()
    
    # Compiler
    from os import system
    system("pdflatex --interaction=batchmode --output-directory=" + dir +' ' + dir +  pdfname + ".tex") 
    system("latexmk -c --interaction=batchmode --output-directory=" + dir + ' ' + dir + pdfname +'.tex' )
    f.close()

    return

tex_display(192, 9, 2, 7, 3, 'AES%dRK_%dr_ENC_r%d_Meet_r%d_KEY_r%d' % (192,9,2,7,3), 0, 20, dir='./milestones/192-9273/old_rule/')