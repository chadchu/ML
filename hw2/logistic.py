import sys
import csv
import numpy as np

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def gradient(D, w, b, y):
	predict = D.dot(w) + b
	predict = sigmoid(predict)
	err = predict - y
	return err.T.dot(D), np.sum(err)

Data = []
label = []
with open(sys.argv[1], 'r') as f:
	fcsv = csv.reader(f)
	next(fcsv)
	for row in fcsv:
		Data.append(row)

Data = np.array(Data, dtype=float).T
u = np.zeros(6)
sigma = np.zeros(6)
for i in range(106):
	if i < 6 and i != 2:
		u[i] = Data[i].mean()
		sigma[i] = Data[i].std()
		Data[i] = (Data[i] - u[i]) / sigma[i]

for i in range(6):
	if i == 2: continue
	Data = np.vstack((Data, Data[i]**2))
	Data = np.vstack((Data, Data[i]**3))

Data = Data.T

with open(sys.argv[2], 'r') as f:
	fcsv = csv.reader(f)
	for row in fcsv:
		label.append(row[0])
label = np.array(label, dtype=float)

epoch = 9001
eta = 0.02
lda = 0.001
w = np.array([ 0.0 for i in range(116) ])
b = 0.0
w_grad_sum = np.array([ 0.0 for i in range(116) ])
b_grad_sum = 0.0

for t in range(epoch):

	w_grad, b_grad = gradient(Data, w, b, label)

	w_grad_sum = w_grad_sum + w_grad**2
	b_grad_sum = b_grad_sum + b_grad**2

	w = (1 - eta * lda) * w - (w_grad * eta) / (w_grad_sum**0.5)
	b = (1 - eta * lda) * b - (b_grad * eta) / (b_grad_sum**0.5)

file = open(sys.argv[4], 'w')
file.write("id,label\n")

cnt = 1

with open(sys.argv[3], 'r') as f:
	fcsv = csv.reader(f)
	next(fcsv)
	for row in fcsv:
		tmp = []
		for i in range(106):
			if i < 6 and i != 2:
				tmp.append( (float(row[i])-u[i])/sigma[i] )
			else:
				tmp.append(row[i])
		for i in range(6):
			if i == 2: continue
			tmp.append( float(tmp[i])**2 )
			tmp.append( float(tmp[i])**3 )

		tmp = np.array(tmp, dtype=float)
		if( sigmoid(tmp.T.dot(w) + b) > 0.5 ):
			file.write('%d,1\n' % cnt)
		else:
			file.write('%d,0\n' % cnt)
		cnt = cnt + 1

file.close()
