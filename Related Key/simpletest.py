import numpy as np
k='str1'
print(type(k))
print(int(k[3])+1)
k+='2'
print(k)
solFile = open('x1.sol', 'r')
K = np.ndarray(shape=(7,4,4), dtype = list)
print(K)
for l in solFile:
    xl = str(l)
    if xl.startswith('K_b'):
        print(xl)
        r = int(xl[4])
        i = int(xl[6])
        j = int(xl[8])
        print(r,i,j, xl[11])
        K[r,i,j] = [xl[11]]
        print(type(K[r,i,j]))
print(K)