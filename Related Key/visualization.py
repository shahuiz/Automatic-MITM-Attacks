import gurobipy as gp
from gurobipy import GRB
import numpy as np
import re

# AES parameters
NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)

TAB = ' '*4

# variable declaration
#total_round = 8 # total round
#start_round = 4   # start round, start in {0,1,2,...,total_r-1}
#match_round = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid
#key_start_round = 4 # key start round

#filename =  'model_4x4_8R_Start_r4_Meet_r1_RelatedKey'
#fnp = './runlog/model_4x4_8R_Start_r4_Meet_r1_RelatedKey.sol'

def color(b,r):
    if b==1 and r==0:
        return 'b'
    if b==0 and r==1:
        return 'r'
    if b==1 and r==1:
        return 'g'
    if b==0 and r==0:
        return 'w'

def writeSol(m: gp.Model, path):
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
            m.write(path + m.modelName + '.sol')
    else:
        print('infeasible')

def displaySol(m:gp.Model, path):
    solFile = open(path + m.modelName +'.sol', 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
    
    match = re.match(r'(\d+)R_ENC_r(\d+)_KEY_r(\d+)Meet_r(\d+)', m.modelName)
    total_round, enc_start, key_start, match_round = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))

    fwd = []    # forward rounds
    bwd = []    # backward rounds

    if enc_start < match_round:
        fwd = list(range(enc_start, match_round))
        bwd = list(range(match_round + 1, total_round)) + list(range(0, enc_start))
    else:
        bwd = list(range(match_round + 1, enc_start))
        fwd = list(range(enc_start, total_round)) + list(range(0, match_round))

    SB_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    SB_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    AK_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    AK_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    KEY_b= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    KEY_r= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    cost_xor = np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    cost_fwd = np.ndarray(shape=(total_round, NCOL), dtype=int)
    cost_bwd = np.ndarray(shape=(total_round, NCOL), dtype=int)

    ini_enc_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_enc_g = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_b = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_r = np.ndarray(shape=(NROW, NCOL), dtype=int)
    ini_key_g = np.ndarray(shape=(NROW, NCOL), dtype=int)

    for r in range(total_round):
        for i in ROW:
            for j in COL:
                SB_b[r,i,j]=Sol["S_b[%d,%d,%d]" %(r,i,j)]
                SB_r[r,i,j]=Sol["S_r[%d,%d,%d]" %(r,i,j)]
                AK_b[r,i,j]=Sol["A_b[%d,%d,%d]" %(r,i,j)]
                AK_r[r,i,j]=Sol["A_r[%d,%d,%d]" %(r,i,j)]

    for ri in range(total_round):
        for i in ROW:
            for j in COL:
                MC_b[ri, i, j] = SB_b[ri, i, (j + i)%NCOL]
                MC_r[ri, i, j] = SB_r[ri, i, (j + i)%NCOL]

    for r in range(total_round+1):
        for i in ROW:
            for j in COL:
                KEY_b[r,i,j]=Sol["K_b[%d,%d,%d]" %(r,i,j)]
                KEY_r[r,i,j]=Sol["K_r[%d,%d,%d]" %(r,i,j)]
                cost_xor[r,i,j] = Sol["Cost_XOR[%d,%d,%d]" %(r,i,j)]
    
    for r in range(total_round):
        for j in COL:
            cost_fwd[r,j] = Sol["Cost_fwd[%d,%d]" %(r,j)]
            cost_bwd[r,j] = Sol["Cost_bwd[%d,%d]" %(r,j)]

    for i in ROW:
        for j in COL:
            ini_enc_b[i,j] = Sol["S_ini_b[%d,%d]" %(i,j)]
            ini_enc_r[i,j] = Sol["S_ini_r[%d,%d]" %(i,j)]
            ini_enc_g[i,j] = Sol["S_ini_g[%d,%d]" %(i,j)]
            ini_key_b[i,j] = Sol["K_ini_b[%d,%d]" %(i,j)]
            ini_key_r[i,j] = Sol["K_ini_r[%d,%d]" %(i,j)]
            ini_key_g[i,j] = Sol["K_ini_g[%d,%d]" %(i,j)]

    ini_df_enc_b = np.sum(ini_enc_b[:,:]) - np.sum(ini_enc_g[:,:])
    ini_df_enc_r = np.sum(ini_enc_r[:,:]) - np.sum(ini_enc_g[:,:])

    ini_df_key_b = np.sum(ini_key_b[:,:]) - np.sum(ini_key_g[:,:])
    ini_df_key_r = np.sum(ini_key_r[:,:]) - np.sum(ini_key_g[:,:])

    DF_b = Sol["DF_b"]
    DF_r = Sol["DF_r"]
    Match = Sol["Match"]
    Obj = Sol["Obj"]

    with open(path + 'Vis_' + m.modelName +'.txt', 'w') as f:
        f.write('ENC FWD: ' + str(ini_df_enc_b) + '\n' + 'ENC BWD: ' + str(ini_df_enc_r) + '\n')
        f.write('KEY FWD: ' + str(ini_df_key_b) + '\n' + 'ENC BWD: ' + str(ini_df_key_r) + '\n')
        f.write('Obj= min{DF_b=%d, DF_r=%d, Match=%d} = %d' %(DF_b, DF_r, Match, Obj) + '\n')
        f.write('\n'*3)

        for r in range(total_round):
            x = ''
            
            if r in fwd:
                x+= '--->'
            elif r in bwd:
                x+= '<---'

            if r == enc_start:
                x+=TAB+'ENC '
            elif r == key_start:
                x+=TAB+'KEY '
            elif r == match_round:
                x+=TAB+'-><-MAT '
            else:
                x+=TAB*2
            
            if r == key_start:
                x+= TAB*5 +'KEY '
            f.write("r%d  " %r + x + '\n')
            f.write('SB#%d'%r +TAB*2+'MC#%d' %r +TAB*2+'AK#%d' %r +TAB*2+'K#%d ' %r +'\n')
            for i in ROW:
                SB = ''
                MC = ''
                AK = ''
                KEY = ''
                
                for j in COL:
                    SB+=color(SB_b[r,i,j], SB_r[r,i,j])
                    MC+=color(MC_b[r,i,j], MC_r[r,i,j])
                    KEY+=color(KEY_b[r,i,j], KEY_r[r,i,j])
                    if r == total_round -1 or r in bwd:
                        AK+=' '

                    else:
                        AK+=color(AK_b[r,i,j], AK_r[r,i,j])
                
                f.write(SB+TAB*2+MC+TAB*2+AK+TAB*2+KEY+'\n')   
            
            f.write('CostDF fwd: '+ str(cost_fwd[r,:]) + '\n')
            f.write('CostDF XOR: ' + '\n' + str(cost_xor[r,:,:]) + '\n')
            f.write('CostDF bwd: '+ str(cost_bwd[r,:]) + '\n')

            f.write('\n'*3)

        # process whiten key
        r = -1
        f.write("r%d  " %r + '\n')
        f.write(6*TAB +'AT  '+ TAB*2 + 'K#-1' + '\n')
        for i in ROW:
            KEY = ''
            AT = ''
            for j in COL:
                KEY+=color(KEY_b[r,i,j], KEY_r[r,i,j])
                AT +=color(AK_b[r+total_round,i,j], AK_r[r+total_round,i,j])
            f.write(6*TAB + AT+ TAB*2 + KEY + '\n')
        
        f.write('CostDF fwd: '+ str(cost_fwd[r+total_round,:]) + '\n')
        f.write('CostDF XOR: ' + '\n' + str(cost_xor[r+total_round,:,:]) + '\n')
        f.write('CostDF bwd: '+ str(cost_bwd[r+total_round,:]) + '\n')

    return 
