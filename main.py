from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import cv2
import numpy as np

from keras import backend as K
K.set_image_dim_ordering('tf')
'''
image_dim_ordering in 'th' mode the channels dimension (the depth) is at index 1 (e.g. 3, 256, 256). " \
"In 'tf' mode is it at index 3 (e.g. 256, 256, 3).
'''


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def get_classes():
    with open('./data/model/synset_words.txt', 'r') as f: 
        lines = f.readlines()
    return np.array(lines)

def parse_args():

    import os
    import argparse

    description = 'Classify an image with vgg16.'
    prog = os.path.basename(__file__)
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument('-i', '--input', type=str, default='data/image/zebra.jpg', help='Picture to classify.')
    args = parser.parse_args()
    assert os.path.exists(args.input), "File '%s' doesn't exist." % (args.input)
    return args


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    args = parse_args()

    l, w = 224, 224

    im_original = cv2.imread(args.input)
    im = cv2.resize(im_original, (224, 224))
    im = im.astype(np.float32)
    # normalization
    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('./data/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)

    # descend sort prediction, and find top-3 classes
    ordered_index = np.argsort(-out)[0]
    all_classes = get_classes()
    classes = '\n'.join(all_classes[ordered_index[0:3]])

    plt.figure(figsize=(10, 6))
    plt.subplot(121, xticks=[], yticks=[], frameon=False)
    plt.imshow(im_original)
    plt.subplot(122, xticks=[], yticks=[], frameon=False)
    plt.text(0.5, 0.5, classes, ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.show()
