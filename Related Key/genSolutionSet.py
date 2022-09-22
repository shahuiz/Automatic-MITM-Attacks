import numpy as np
import time
import sys 
import AES_RkMitM_zty_wLastRoundMatch as RkMitM

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

file = open("Related Key/runlog.txt","w")
file.close()

for enc in range(8):
    continue
    for mat in range(8):
        if mat != 7:
            continue

enc = 4
mat = 1

for key in range(-1,8):
    f = open('Related Key/runlog.txt', 'a')
    start = time.time()
    msg = RkMitM.solve(key_size=128, total_round=8, start_round=enc, key_start_round=key, match_round=mat)
    end = time.time()
    f.write(msg + '\n' + TAB + 'time cost: ' + '%.2f' % (end - start) + '\n')

    f.close()