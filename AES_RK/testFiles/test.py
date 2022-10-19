import numpy as np
a = np.asarray([[0,0],[1,0],[2,0],[3,0]])
print(np.roll(a[:,0],-1))

#array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
#>>> np.roll(a,-2)
#array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])