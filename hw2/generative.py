import sys
import csv
import numpy as np
Data = []
label = []
with open(sys.argv[1], 'r') as f:
	fcsv = csv.reader(f)
	next(fcsv)
	for row in fcsv:
		Data.append(row)
Data = np.array(Data, dtype=float)
with open(sys.argv[2], 'r') as f:
	fcsv = csv.reader(f)
	for row in fcsv:
		label.append(row[0])
label = np.array(label, dtype=float)

class1 = []
class2 = []
n1 = 0
n2 = 0

for i in range(Data.shape[0]):
	if label[i] == 1:
		class1.append(list(Data[i]))
		n1 += 1
	if label[i] == 0:
		class2.append(list(Data[i]))
		n2 += 1

class1 = np.array(class1, dtype=float).T
class2 = np.array(class2, dtype=float).T

u1 = np.zeros(106)
for i in range(106):
	u1[i] = class1[i].mean()
u2 = np.zeros(106)
for i in range(106):
	u2[i] = class2[i].mean()

sigma1 = np.zeros((106, 106))
sigma2 = np.zeros((106, 106))

class1 = class1.T
class2 = class2.T

for i in range((class1.shape)[0]):
	tmp = class1[i] - u1
	sigma1 = sigma1 + tmp.reshape(106,1).dot(tmp.reshape(1,106))

for i in range((class2.shape)[0]):
	tmp = class2[i] - u2
	sigma2 = sigma2 + tmp.reshape(106,1).dot(tmp.reshape(1,106))

sigma = (n1 * sigma1 + n2 * sigma2) / (n1 + n2)

w = (u1 - u2).dot(np.linalg.inv(sigma))
b = 0.5*(((u2.reshape(1,106)).dot(np.linalg.inv(sigma)))).dot(u2.reshape(106,1)) - 0.5*(((u1.reshape(1,106)).dot(np.linalg.inv(sigma)))).dot(u1.reshape(106,1))
b = b[0,0] + np.log(n1/n2)

cnt = 0
for i in range(32561):
	k = Data[i].dot(w) + b
	k = 1 / (1 + np.exp(-k))
	if(k > 0.5 and label[i] == 1):
		cnt = cnt + 1
	elif(k <= 0.5 and label[i] == 0):
		cnt = cnt + 1

file = open(sys.argv[4], 'w')
file.write("id,label\n")

cnt = 1

with open(sys.argv[3], 'r') as f:
	fcsv = csv.reader(f)
	next(fcsv)
	for row in fcsv:
		tmp = np.array(row, dtype=float)
		if( 1/(1+np.exp(-(tmp.T.dot(w) + b))) > 0.5 ):
			file.write('%d,1\n' % cnt)
		else:
			file.write('%d,0\n' % cnt)
		cnt = cnt + 1

file.close()
