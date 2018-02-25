import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
import numpy as np
import csv

def main():
    emotion_classifier = load_model('model.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['conv2d_2']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]
    # print(name_ls)
    pxl = []
    with open('train.csv', 'r') as f:
        fcsv = csv.reader(f)
        for i in range(10):
            next(fcsv)
        a = next(fcsv)
        pxl = a[1]
    pxl = np.fromstring(pxl, dtype=float, sep=' ').reshape((1, 48, 48, 1))

    private_pixels = []
    private_pixels.append(pxl)

    choose_id = 17
    photo = private_pixels[0]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        # fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        fig.show()
        img_path = os.path.join('.', 'filter')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

if __name__ == '__main__':
	main()
