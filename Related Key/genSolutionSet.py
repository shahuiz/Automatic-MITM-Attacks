import numpy as np
import time
import os
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

file = open("Related Key/runlog.txt","w")
file.close()

dir = './128RK_output/'
if not os.path.exists(path= dir):
    os.makedirs(dir)

total_start = time.time()
for enc in range(8):
    for mat in range(8):
        for key in range(-1,8):
            f = open('Related Key/runlog.txt', 'a')
            start = time.time()
            Params, Sol = RkMitM.solve(key_size=128, total_round=8, start_round=enc, key_start_round=key, match_round=mat, dir=dir)
            end = time.time()
            f.write('total: %d start: %d meet :%d key_start: ' % Params + TAB + 'time cost: ' + '%.2f' % (end - start) + Sol)
total_end = time.time()

f.write('\n'*2 + 'total search time: ' + '%.2f' % (total_end - total_start) + '\n')         
f.close()