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

def displaySol(key_size:int, total_round:int, enc_start_round:int, match_round:int, key_start_round:int, model_name:str, sol_i:int, dir):
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
        if len(W_x.shape) > 2:
            nrow = len(W_x[0])
            ncol = len(W_x[0][0])
            for ri in range(nrow):
                i = nrow - 1 - ri
                for j in range(ncol):
                    file.write(color_fill[W_x[r,i,j], W_y[r,i,j]] + ' (%d,%d) rectangle + (1,1);\n'   %(j, i))
        else:
            nrow = 4
            ncol = 1
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



    color_fill = np.ndarray(shape=(2, 2),dtype='object')
    color_fill[0, 0] = '\\fill[\\UW]'
    color_fill[0, 1] = '\\fill[\\BW]'
    color_fill[1, 0] = '\\fill[\\FW]'
    color_fill[1, 1] = '\\fill[\\CW]'
    
    if NROW == 4:
        y_shift = NROW
        x_shift = NCOL
    else:
        y_shift = NROW
        x_shift = NCOL // 2

    

    outfile = './x'
    f = open(outfile + '.tex', 'w')
    
    # write latex header
    f.write( '%' + ' Vis_' + model_name + '\n'
        '\\documentclass{standalone}' + '\n'
        '\\usepackage[usenames,dvipsnames]{xcolor}' + '\n'
        '\\usepackage{amsmath,amssymb,mathtools,tikz,calc,pgffor}' + '\n'
        '\\usepackage{xspace}' + '\n'
        '\\usetikzlibrary{crypto.symbols,patterns,calc}' + '\n'
        '\\tikzset{shadows=no}' + '\n'
        '\\input{macro}' + '\n')
    f.write('\n\n')
    
    f.write( '%' + 'document starts' + '\n'
        '\\begin{document}' + '\n' +
        '\\begin{tikzpicture}[scale=0.2, every node/.style={font=\\boldmath\\bf}]' + '\n'
	    '\\everymath{\\scriptstyle}' + '\n'
	    '\\tikzset{edge/.style=->, >=stealth, arrow head=8pt, thick};' + '\n')
    f.write('\n\n')

    # draw solution
    for r in range(total_round):
        mc_cost_fwd_col = 0
        mc_cost_bwd_col = 0
        for i in COL:
            mc_cost_fwd_col += mc_cost_fwd[r, i]
            mc_cost_bwd_col += mc_cost_fwd[r, i]
        
        if r in fwd or r == match_round:
            arrow = '->'
        else:
            arrow = '<-'

        ## SubByte state
        slot = 0
        f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
        draw_cells(SB_x, SB_y, f)
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        # write enc start sign
        if r == enc_start_round:
            f.write('\\path (' + str(NCOL//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_df_enc_b) + '~\\DoFF,~+' + str(ini_df_enc_r) + '~\\DoFB)$};'+'\n')
            f.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')

        # MixCol state
        slot += 1
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
        draw_cells(MC_x, MC_y, f)
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%d$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny SupP}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
        
        # SupP MixCol state
        slot += 1
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
        draw_cells(fMC_x, fMC_y, f)
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        if r != total_round -1:
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        else:
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Add}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift*3))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
        
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
        draw_cells(bMC_x, bMC_y, f)
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\MC^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        if r != total_round - 1:
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        else: 
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Add}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift*3))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
            
        # SupP AddKey state
        slot += 1
        if r != total_round - 1:
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            draw_cells(fAK_x, fAK_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\AK^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Add}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
            
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
            draw_cells(bAK_x, bAK_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\AK^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Add}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
        
        # SupP SubByte state (next round)
        slot += 1
        if r != total_round - 1:
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            original_r = copy.deepcopy(r)
            r = (r+1)%total_round
            draw_cells(fSB_x, fSB_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
            r = original_r
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Eval}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
            
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
            original_r = copy.deepcopy(r)
            r = (r+1)%total_round
            draw_cells(bSB_x, bSB_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
            r = original_r
            #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
        
        else: 
            # AT state
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            draw_cells(fAK_x, fAK_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\AT$};\n'    %(NCOL//2, NROW+0.5))
            #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Add}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
            
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
            draw_cells(bAK_x, bAK_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\AT$};\n'    %(NCOL//2, NROW+0.5))
            #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Add}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
        
        # SubByte state (next round)
        slot += 1
        if r != total_round - 1:
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            original_r = copy.deepcopy(r)
            r = (r+1)%total_round
            draw_cells(SB_x, SB_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, r))
            r = original_r
            #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Eval}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
        
        # SupP key state
        slot += 1
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
        draw_cells(fKEY_x, fKEY_y, f)
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
        
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
        draw_cells(bKEY_x, bKEY_y, f)
        draw_gridlines(f)
        f.write('\\path (%f,%f) node {\\scriptsize$\\K^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')

        if r == total_round - 1:
            # whitening key
            r = -1
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-total_round*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            draw_cells(fKEY_x, fKEY_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\K^{w}F$};\n'    %(NCOL//2, NROW+0.5))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
            
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-total_round*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
            draw_cells(bKEY_x, bKEY_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\K^{w}B$};\n'    %(NCOL//2, NROW+0.5))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')

            # SupP SB0
            slot -= 2
            r = 0
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-total_round*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            draw_cells(fSB_x, fSB_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
            f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny Eval}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
            
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-total_round*(3*NROW+y_shift)-1.5*NROW, slot*(NCOL+x_shift))) 
            draw_cells(bSB_x, bSB_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
            #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny MC}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')

            # SB0
            slot += 1
            f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-total_round*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            draw_cells(SB_x, SB_y, f)
            draw_gridlines(f)
            f.write('\\path (%f,%f) node {\\scriptsize$\\SB^%d$};\n'    %(NCOL//2, NROW+0.5, r))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')
            
            break
            
        continue

    # draw key schedule
    for r in range(Nr):
        slot = 7
        # SupP Key Schedule 
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(Nk+x_shift))) 
        draw_cells(fKSch_x, fKSch_y, f)
        draw_gridlines(f, 'KS')
        f.write('\\path (%f,%f) node {\\scriptsize$\\KS^%dF$};\n'    %(NCOL//2, NROW+0.5, r))
        #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny KeySchedule}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')
        
        f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift)-1.5*NROW, slot*(Nk+x_shift))) 
        draw_cells(bKSch_x, bKSch_y, f)
        draw_gridlines(f, 'KS')
        f.write('\\path (%f,%f) node {\\scriptsize$\\KS^%dB$};\n'    %(NCOL//2, NROW+0.5, r))
        #f.write('\\draw[edge, %s] (%f,%f) -- node[above] {\\tiny KeySschedule}+(%d,0);\n'    %(arrow, NCOL, NROW//2, x_shift))
        f.write('\n'+'\\end{scope}'+'\n')
        f.write('\n\n')

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

        continue
    
    
    ## Final footnote
    f.write('\\begin{scope}[yshift = %f cm, xshift = %f cm]\n\n'   %(-(total_round+2)*(3*NROW+y_shift), 2*(NCOL+x_shift))) 
    #f.write('\\begin{scope}[yshift =' + str(- total_round * (NROW + y_shift) + y_shift)+' cm, xshift =' +str(2 * (NCOL + x_shift))+' cm]'+'\n')
    f.write(
        '\\node[draw, thick, rectangle, text width=6.5cm, label={[shift={(-2.8,-0)}]\\footnotesize Config}] at (-7, 0) {' + '\n'
	    '{\\footnotesize' + '\n'
	    '$\\bullet~(\\varInitBL,~\\varInitRD)~=~(+' + str(ini_df_enc_b) + '~\\DoFF,~+' + str(ini_df_enc_r) + '~\\DoFB)~$' + '\\\ \n'
	    '$\\bullet~(\\varDoFBL,~\\varDoFRD,~\\varDoM)~=~(+' + 
        str(int(DF_b)) + '~\\DoFF,~+' + 
        str(int(DF_r)) + '~\\DoFB,~+' + 
        str(int(Match)) + '~\\DoM)$' + '\n'
	    '}' + '\n'
	    '};' + '\n'
        )
    f.write('\n'+'\\end{scope}'+'\n')
    
    
    f.write('\n\n')
    f.write('\\end{tikzpicture}'+'\n\n'+'\\end{document}')
    f.close()
    from os import system
    system("pdflatex -output-directory='./' ./" + outfile + ".tex") 
    
    f.close()

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