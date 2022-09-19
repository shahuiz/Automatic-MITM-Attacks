from curses import KEY_A1
import gurobipy as gp
from gurobipy import GRB
import numpy as np
#import result_vis as vis

# AES parameters
NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)

# variable declaration
total_round = 8 # total round
start_round = 4   # start round, start in {0,1,2,...,total_r-1}
match_round = 1  # meet in the middle round, mid in {0,1,2,...,total_r-1}, start != mid
key_start_round = 4 # key start round

fnp = './runlog/model_4x4_8R_Start_r4_Meet_r1_RelatedKey.sol'

def color(b,r):
    if b==1 and r==0:
        return 'b'
    if b==0 and r==1:
        return 'r'
    if b==1 and r==1:
        return 'g'
    if b==0 and r==0:
        return 'w'


def printSol(outfile):
    solFile = open(outfile, 'r')
    Sol = dict()
    for line in solFile:
        if line[0] != '#':
            temp = line
            temp = temp.split()
            Sol[temp[0]] = round(float(temp[1]))
    
    SB_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    SB_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    MC_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    AK_b = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    AK_r = np.ndarray(shape=(total_round, NROW, NCOL), dtype=int)
    KEY_b= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)
    KEY_r= np.ndarray(shape=(total_round+1, NROW, NCOL), dtype=int)

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
    
    with open('test.out', 'w') as f:
        f.write('SBSR'+'    '+'MC  '+'    '+'AK  '+'    '+'KEY  '+'    '+'\n')
        for r in range(total_round):
            f.write("round %d\n" %r)
            for i in ROW:
                SB = ''
                MC = ''
                AK = ''
                KEY = ''
                for j in COL:
                    SB+=color(SB_b[r,i,j], SB_r[r,i,j])
                    MC+=color(MC_b[r,i,j], MC_r[r,i,j])
                    AK+=color(AK_b[r,i,j], AK_r[r,i,j])
                    KEY+=color(KEY_b[r,i,j], KEY_r[r,i,j])
                
                f.write(SB+'    '+MC+'    '+AK+'    '+KEY+'    '+'\n')   
            f.write('\n'*3)

    
    print(Sol)
    return 

printSol(fnp)