from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import csv
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

mylabel = []
label = []

a = open('ans.csv', 'r')
b = open('n.csv', 'r')
fa = csv.reader(a)
fb = csv.reader(b)
next(a)
cnt = 0
for i in range(7178):
	m = next(fa)
	n = next(fb)
	mylabel.append(m[1])
	label.append(n[0])

np.set_printoptions(precision=2)

mylabel = np.array(mylabel, dtype=int)
label = np.array(label, dtype=int)

conf_mat = confusion_matrix(label, mylabel)
print(conf_mat)

plt.figure()
plot_confusion_matrix(conf_mat, classes=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
plt.show()
