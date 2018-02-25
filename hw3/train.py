import numpy as np
import csv
import h5py
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

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

T = []
for i in range(48):
	tmp = [0]*48
	tmp[47-i] = 1
	T.append(tmp)
T = np.array(T)

Data = []
label = []
with open(sys.argv[1], 'r') as f:
	fcsv = csv.reader(f)
	next(fcsv)
	for row in fcsv:
		label.append( row[0] )
		label.append( row[0] )
		a = np.array(row[1].split(), dtype=int)
		a = a.reshape(48,48)
		a = equalize(a)
		Data.append(a)
		Data.append(a.dot(T))

	label = np.array(label)
	Data = np.array(Data)
	Data = Data.reshape(28709*2, 48, 48, 1)
print('Done\n')
label = np_utils.to_categorical(label, 7)

model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(.2))
model.add(AveragePooling2D((2,2)))
model.add(Dropout(.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Dropout(.2))
model.add(AveragePooling2D((2,2)))
model.add(Dropout(.25))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Dropout(.4))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.4))

model.add(Flatten())

model.add(Dense(units=1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=7,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(Data,label,batch_size=256,epochs=100,validation_split=.2)

model.save('mymodel.h5')

score = model.evaluate(Data,label)
print ('\nTrain Acc:', score[1])

del model
del Data
del label
