import numpy as np
import pandas as pd
import sys
import os
from keras.models import load_model

model = load_model('hw6.h5')

test = pd.read_csv(os.path.join(sys.argv[1], 'test.csv')).values
X1 = test[:, 1]
X2 = test[:, 2]

Y = model.predict([X1, X2])

f = open(sys.argv[2], 'w')
f.write('TestDataId,Rating\n')
for i in range(100336):
    f.write('%d,%f\n'%(i+1,Y[i,0]))
