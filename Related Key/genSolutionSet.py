# parameter to change
key_size = 128
search_round = 8

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

if key_size > 128:
    key_search_round = search_round - 1
else:
    key_search_round = search_round

dir = './' + str(key_size) +'_'+ str(search_round)+ 'r_RK_output/'
if not os.path.exists(path= dir):
    os.makedirs(dir)

logdir = dir + str(key_size) + '_' + str(search_round) +'r_runlog.txt'
file = open(logdir,"w")
file.close()

sol_count = 0
total_start = time.time()
for enc in range(search_round):
    for mat in range(search_round):
        for key in range(-1, key_search_round):
            f = open(logdir, 'a')
            start = time.time()
            Params, flag, Sol = RkMitM.solve(key_size=key_size, total_round=search_round, start_round=enc, key_start_round=key, match_round=mat, dir=dir)
            end = time.time()
            f.write('total: %d start: %d meet: %d key_start: %d' % Params + TAB + 'time cost: ' + '%.2f' % (end - start) + TAB + Sol + '\n')
            sol_count += flag
total_end = time.time()

f.write('\n'*2 + 'total search time: ' + '%.2f' % (total_end - total_start) + '\n') 
f.write('solution count: ' + '%d'%sol_count)        
f.close()