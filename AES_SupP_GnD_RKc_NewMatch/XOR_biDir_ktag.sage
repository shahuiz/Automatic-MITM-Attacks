poly = Polyhedron(vertices= [
    #in1, in2, out, k, cf,cb
    (0,0, 0,0, 0,0, 0, 0, 0),
    (0,0, 0,1, 0,0, 0, 0, 0),
    (0,0, 1,0, 0,0, 0, 0, 0),
    (0,0, 1,1, 0,0, 0, 0, 0),
    
    (0,1, 0,0, 0,0, 0, 0, 0),
    (0,1, 0,1, 0,1, 1, 0, 0), (0,1, 0,1, 1,1, 1, 0, 1),
    (0,1, 1,0, 0,0, 1, 0, 0),
    (0,1, 1,1, 0,1, 1, 0, 0),
    
    (1,0, 0,0, 0,0, 0, 0, 0),
    (1,0, 0,1, 0,0, 1, 0, 0),
    (1,0, 1,0, 1,0, 1, 0, 0), (1,0, 1,0, 1,1, 1, 1, 0),
    (1,0, 1,1, 1,0, 1, 0, 0),
    
    (1,1, 0,0, 0,0, 0, 0, 0),
    (1,1, 0,1, 0,1, 1, 0, 0),
    (1,1, 1,0, 1,0, 1, 0, 0),
    (1,1, 1,1, 1,1, 1, 0, 0)    
])

A = []
B = []
for line in poly.inequality_generator():
    strline = str(line)
    strline = strline.replace('(', '|')
    strline = strline.replace(')', '|')
    strline = strline.replace('+', '|')
    strline = strline.replace('=', '|')
    strline = strline.split('|')
    A.append(strline[1].split(', '))
    B.append(strline[3][1])

#print(A, B)
intA =[]
for x in A:
    temp = []
    for y in x:
         temp.append(int(y))
    intA.append(temp)
print(intA)

intB=[]
for x in B:
    intB.append(int(x))
print(intB)

print(len(intA), len(intB))