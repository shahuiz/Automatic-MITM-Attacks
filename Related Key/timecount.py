solFile = open('Related Key/runlog.txt', 'r')
total_time = 0
for line in solFile:
    if line[0] == ' ':
        temp = line
        temp = temp.split(": ")
        temp = temp[1].split("\n")
        total_time +=float(temp[0])
print(total_time)