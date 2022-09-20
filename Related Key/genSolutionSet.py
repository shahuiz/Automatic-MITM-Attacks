import numpy as np
import sys 
import AES_RkMitM_zty as RkMitM

NROW = 4
NCOL = 4
NGRID = NROW * NCOL
NBRANCH = NROW + 1     # AES MC branch number
ROW = range(NROW)
COL = range(NCOL)
TAB = ' '*4

# linear constriants for XOR operations
XOR_A = np.asarray([
    [0, 0, 0, 0, 0, 0, 1],
    [-1, 0, -1, 0, 1, 0, -2],
    [0, 0, 1, 0, -1, 0, 1],
    [0, -1, 0, -1, 0, 1, 0],
    [0, 0, 0, 1, 0, -1, 0],
    [1, 0, 0, 0, -1, 0, 1],
    [0, 1, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 1, 0, -1],
    [0, 0, 0, 0, 0, 1, -1]])

XOR_B = np.asarray([0,1,0,1,0,0,0,0,0])

f = open('./runlog.txt', "a")
for enc in range(7):
    for key in range(8):
    #key=enc
        for mat in range(7):
            f = open('runlog.txt', 'a')
            msg = RkMitM.solve(key_size=128, total_round=8, start_round=enc, key_start_round=key, match_round=mat)
            f.write(msg)
            f.write('\n')
            #print(msg, '\n')
            f.close()