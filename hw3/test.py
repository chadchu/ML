import numpy as np
import csv
import sys
import h5py
from keras.models import load_model

def equalize(im):
	ret = np.array(im, dtype=float)
	if ret.sum() == 0:
		return ret
	l = im.shape[0]
	h = [0] * 256
	for i in range(l):
		for j in range(l):
			h[ im[i][j] ] += 1
	h = np.array(h, dtype=float)
	cum = [ sum(h[:i+1]) for i in range(256) ]
	for i in cum:
		if i != 0:
			cdfmin = i
			break
	for i in range(l):
		for j in range(l):
			ret[i][j] = round( (cum[im[i][j]]-cdfmin)*255.0 / float(l*l-cdfmin)  )
	return ret

Data = []
with open(sys.argv[1], 'r') as f:
	fcsv = csv.reader(f)
	next(fcsv)
	for row in fcsv:
		img = row[1].split()
		img = np.array(img, dtype=int).reshape(48,48)
		Data.append( equalize(img) )
	Data = np.array(Data, dtype=float)
	Data = Data.reshape(Data.shape[0], 48, 48, 1)
model = load_model('model.h5')
y = model.predict_classes(Data)
f = open(sys.argv[2], 'w')
f.write('id,label\n')
for i in range(y.shape[0]):
	f.write('%d,%d\n' % (i, y[i]))
f.close()
