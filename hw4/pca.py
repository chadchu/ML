from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

s = 'ABCDEFGHIJ'
imgs = []
avg = np.zeros((64,64))
for i in range(10):
    for j in range(10):
        img = Image.open('faceExpressionDatabase/%s%02d.bmp' % (s[i], j))
        avg += np.array(img)
        imgs.append(np.array(img).flatten())

avg /= 100
Image.fromarray(avg.astype(np.uint8)).save('avg.png')
avg = avg.astype(np.uint8).flatten()

imgs = np.array(imgs, dtype=float)
U, s, V = np.linalg.svd(imgs-imgs.mean(axis=0), full_matrices=False)
S = np.diag(s)

eigen_faces = V

fig = plt.figure(figsize=(3,3))

for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(eigen_faces[i].reshape((64,64)), cmap='gray')
    ax.axis("off")

fig.savefig('ef.png', bbox_inches='tight')

original = plt.figure(figsize=(10,10))
for i in range(100):
    ax = original.add_subplot(10, 10, i+1)
    ax.imshow(imgs[i].reshape((64,64)), cmap='gray')
    ax.axis("off")
original.savefig('original.png', bbox_inches='tight')

V5 = V[:5, :]
Xr = (imgs-imgs.mean(axis=0)).dot(V5.T)
X_hat = Xr.dot(V5)
X_hat += imgs.mean(axis=0)
recoverd = plt.figure(figsize=(10,10))
for i in range(100):
    ax = recoverd.add_subplot(10, 10, i+1)
    ax.imshow(X_hat[i].reshape((64,64)), cmap='gray')
    ax.axis("off")
recoverd.savefig('recoverd.png', bbox_inches='tight')

for i in range(1,101):
    xr = (imgs-imgs.mean(axis=0)).dot(V[:i, :].T)
    xh = xr.dot(V[:i, :])
    xh += imgs.mean(axis=0)
    err = (imgs - xh)
    err = err**2
    mean = (err.sum())/409600
    rmse = (mean**0.5)/256
    # print('%d %f' %(i, rmse.mean()))
