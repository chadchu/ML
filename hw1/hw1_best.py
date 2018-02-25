import csv
import numpy as np
import sys
raw_data = []
idx_dict = {}

myidx = 0
f = open(sys.argv[1], "r", encoding = 'big5')
fcsv = csv.reader(f)
next(fcsv)
for row in fcsv:
    if row[2] not in idx_dict:
        idx_dict[ row[2] ] = myidx;
        myidx += 1
        raw_data.append([])
    if row[2] == 'RAINFALL':
        for i in range(3, len(row)):
            if(row[i] == 'NR'):
                raw_data[ idx_dict[ row[2] ] ].append(0.0)
            else:
                raw_data[ idx_dict[ row[2] ] ].append(float(row[i]))
    else:
        for i in range(3, len(row)):
            raw_data[ idx_dict[ row[2] ] ].append(float(row[i]))
f.close()

raw_data = np.array(raw_data)

x = []
y = []

for i in range(12):
    for j in range(471):
        tmp = []
        for k in range(9):
            tmp.append(raw_data[9][i*480+j+k])
        for k in range(5):
            tmp.append(raw_data[8][i*480+j+k+4])
        for k in range(3):
            tmp.append(raw_data[7][i*480+j+k+6])
        for k in range(3):
            tmp.append(raw_data[10][i*480+j+k+6])
        for k in range(2):
            tmp.append(raw_data[16][i*480+j+k+7])
        for k in range(1):
            tmp.append(raw_data[5][i*480+j+k+8])
        for k in range(4):
            tmp.append(float(raw_data[9][i*480+j+k+5])**2)
        for k in range(2):
            tmp.append(float(raw_data[8][i*480+j+k+7])**2)
        x.append(tmp)
        y.append(raw_data[9][i*480+j+9])
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

b = 0
w = np.array([0.0001 for i in range(29)])
lr = 0.005
iteration = 60001

b_lr = 0.00
w_lr = np.array([0.0 for i in range(29)])

for k in range(iteration):
    
    b_grad = 0.0
    w_grad = np.array([0.0 for i in range(29)])
    
    predict = x.dot(w)
    err = y-b-predict
    
    b_grad -= 2.0*(np.sum(predict))
    w_grad -= 2.0*((err.T).dot(x))

    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2

    b = b - (lr * b_grad)/np.sqrt(b_lr)
    w = w - (lr * w_grad)/np.sqrt(w_lr) 

file = open(sys.argv[3], 'w')
file.write("id,value\n")

test_f = open(sys.argv[2], 'r')
test_csv = csv.reader(test_f)

test_data = []

for row in test_csv:
    tmp = []
    if row[1] == 'RAINFALL':
        for i in range(2, len(row)):
            if(row[i] == 'NR'):
                tmp.append('0.0')
            else:
                tmp.append(float(row[i]))
    else:
        for i in range(2, len(row)):
            tmp.append(float(row[i]))

    test_data.append(tmp)

test_data = np.array(test_data, dtype=float)

for i in range(240):

    t = []
    for k in range(9):
        t.append(test_data[i*18+9][k])
    for k in range(5):
        t.append(test_data[i*18+8][k+4])
    for k in range(3):
        t.append(test_data[i*18+7][k+6])
    for k in range(3):
        t.append(test_data[i*18+10][k+6])
    for k in range(2):
        t.append(test_data[i*18+16][k+7])
    for k in range(1):
        t.append(test_data[i*18+5][k+8])
    for k in range(4):
        t.append(float(test_data[i*18+9][k+5])**2)
    for k in range(2):
        t.append(float(test_data[i*18+8][k+7])**2)
    
    t = np.array(t, dtype=float)

    y_predict = ( t.dot(w) + b )
    file.write("id_%d,%f\n" % (i, y_predict))

file.close()