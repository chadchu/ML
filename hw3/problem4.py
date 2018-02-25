from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image

pxl = []
model = load_model('model.h5')
with open('train.csv', 'r') as f:
	fcsv = csv.reader(f)
	for i in range(10):
		next(fcsv)
	a = next(fcsv)
	pxl = a[1]

pxl = np.fromstring(pxl, dtype=float, sep=' ').reshape((1, 48, 48, 1))
input_img = model.input

val_proba = model.predict(pxl)
pred = val_proba.argmax(axis=-1)
target = K.mean(model.output[:, pred])
grads = K.gradients(target, input_img)[0]

grads = (grads-K.mean(grads))/K.std(grads)
grads = (grads-K.min(grads))/(K.max(grads)-K.min(grads))

fn = K.function([input_img, K.learning_phase()], [grads])

heatmap = fn([pxl, False])
heatmap = np.array(heatmap).reshape(48,48)

thres = 0.5
see = pxl.reshape(48, 48)
see[np.where(heatmap <= thres)] = np.mean(see)

plt.figure()
plt.imshow(heatmap, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('heatmap.png')

plt.figure()
plt.imshow(see,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig('masked.png')
