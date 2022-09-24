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

dir = './192RK_output/'
if not os.path.exists(path= dir):
    os.makedirs(dir)

logdir = dir + '192_9r_runlog.txt'
file = open(logdir,"w")
file.close()

sol_count = 0
total_start = time.time()
for enc in range(9):
    for mat in range(9):
        for key in range(-1, 9-1):
            f = open(logdir, 'a')
            start = time.time()
            Params, flag, Sol = RkMitM.solve(key_size=192, total_round=9, start_round=enc, key_start_round=key, match_round=mat, dir=dir)
            end = time.time()
            f.write('total: %d start: %d meet: %d key_start: %d' % Params + TAB + 'time cost: ' + '%.2f' % (end - start) + TAB + Sol + '\n')
            sol_count += flag
total_end = time.time()

f.write('\n'*2 + 'total search time: ' + '%.2f' % (total_end - total_start) + '\n') 
f.write('solution count: ' + '%d'%sol_count)        
f.close()