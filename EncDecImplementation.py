from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator
from pathlib import Path
import tensorflow as tf
import parseXML as pXML
from scipy.special import expit
from tensorflow.keras import layers, Model, utils
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Input, Flatten, LSTM, Lambda, Dense, Add
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
# import tensorflow_gan as tfgan
dataList = np.load('shortdata.npy', allow_pickle=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_eager_execution()


def create_gen_model():

    img_input = layers.Input(shape = (None, 48, 48, 1), name = 'ImagesInputLayer')

    conv1 = TimeDistributed(Conv2D(48, 3, padding = 'same', activation='relu'), name = 'ConvolutionalLayer1')(img_input)
    maxpool1 = TimeDistributed(MaxPooling2D((2,2)), name = 'MaxPoolingLayer1')(conv1)
    conv2 = TimeDistributed(Conv2D(48, 3, padding = 'same', activation='relu'), name = 'ConvolutionalLayer2')(maxpool1)
    maxpool2 = TimeDistributed(MaxPooling2D((2,2)), name = 'MaxPoolingLayer2')(conv2)
    conv3 = TimeDistributed(Conv2D(48, 3, padding = 'same', activation='relu') , name = 'ConvolutionalLayer3')(maxpool2)
    flatten = TimeDistributed(Flatten())(conv3)
    #lstm1 = LSTM(100, return_sequences= True)(flatten)
    #lstm2 = LSTM(500, return_sequences=True)(flatten)
    encoder, e_state_h, e_state_c = LSTM(200, return_state = True, name = 'Encoder')(flatten)
    # encoder_outputs, state_h, state_c = encoder(img_input)
    # encoder_states = [tf.reshape(state_h, (48, 48)), tf.reshape(state_c, (48, 48))]

    pen_input = Input(shape=(None, 3), name = 'PenPointsInputLayer')
    decoder_lstm = LSTM(200, name = 'Decoder')
    decoder_outputs= decoder_lstm(pen_input, initial_state=[e_state_h, e_state_c])

    ## Means and variances for position Gaussians
    mu = Dense(2, activation='linear', name='mu')(decoder_outputs)
    #sigma = Dense(2, activation='relu')(decoder_outputs)
    #gauss = Lambda(Gaussian)(sigma)
    #offsets = Add()([gauss, mu])


    ## Probabilities for Bernoulli's to determine stroke ending and to signify
    ## the end of the phrase.
    p = layers.Dense(2, activation='sigmoid', name = 'p')(decoder_outputs)
    # outputs = layers.Concatenate()([offsets, p])

    model = Model([img_input, pen_input], [mu, p])
    return model

def Gaussian(sigma):
    return np.random.randn(2) * sigma

def Bernoulli(p):
    return np.random.binomial(1, p)


gen_model = create_gen_model()
losses = {'mu': mean_squared_error, 'p': binary_crossentropy}
lossWeights = {'mu': 10.0, 'p': 1.0}
gen_model.compile(optimizer='sgd', loss=losses, loss_weights = lossWeights,
              metrics=['accuracy'])
utils.plot_model(gen_model, 'gen_model.png', show_shapes=True)

# gen_model.load_weights('./checkpoints/checkpoint')
# gen_model.save_weights('./checkpoints/checkpoint')

def random_image_seq(l):
    image_seq = []
    for i in range(l):
        img = np.random.uniform(low=0, high=255, size=(48, 48))
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        image_seq.append(img)
    return image_seq

def random_pen_offs(l):
    pen_seq = []
    for i in range(l):
        point = np.random.uniform(low=-10, high=10, size = 2)
        up = np.random.randint(low=0, high = 2, size = 1)
        pen_point = np.concatenate((point, up))
        pen_seq.append(pen_point)
    return pen_seq

def gen_hw(img_seq, gen_model, maxlength):
    inkpoints = [(0.0, 0.0, 0.0)]
    img_seq = np.asarray(imgSeqs0)
    img_seq_reshaped = np.reshape(img_seq, (1, img_seq.shape[0], img_seq.shape[1], img_seq.shape[2], img_seq.shape[3]))

    stop = False
    while not stop:
        pen_seq = np.asarray(inkpoints)
        pen_seq_reshaped = np.reshape(pen_seq, (1, pen_seq.shape[0], pen_seq.shape[1]))
        (offs, p) = gen_model.predict([img_seq_reshaped, pen_seq_reshaped])
        if p[0][0] >(1/2):
            strokeup = 1
        else:
            strokeup = 0
        inkpoints.append(np.array((offs[0][0], offs[0][1], strokeup)))
        print(len(inkpoints))
        if p[0][1] > 0.8 or len(inkpoints) > maxlength:
            stop = True
    return inkpoints


def draw_offsets(inkpoints):
    start = [0, 0, 0]
    for i in range(len(inkpoints) - 1):
        next = inkpoints[i + 1]

        if start[2] == 0:
            color = 'black'
        else:
            color = 'white'
        plt.plot([start[0], next[0] + start[0]], [-start[1], -(next[1] + start[1])], color=color)
        start[0] = start[0] + next[0]
        start[1] = start[1] + next[1]
        start[2] = next[2]

    plt.show()



(imgSeqs, WL) = pXML.xmltosequencepair(pXML.fn)
imgSeqs0 = imgSeqs[0]
pointSeq0 = WL[0]

# x_t_i, x_t_p, y_t_point, y_t_prob = training_data(imgSeqs0, pointSeq0)

### Enforce that the image sequence lengths are the same
###
def get_training_data(numpoints, numimgs):
    x_train_images = []
    x_train_points = []
    y_train_points = []
    y_train_probs = []
    for j in range(dataList.shape[0]):
        imgSeqs = dataList[j][0]
        pointSeqs = dataList[j][1]
        for k in range(len(imgSeqs)):
            point_seq = pointSeqs[k]
            img_seq = imgSeqs[k]
            if len(img_seq) == numimgs:
                if numpoints == len(point_seq) - 2:
                    x = np.asarray(point_seq[0:numpoints + 1])
                    y = point_seq[numpoints + 1]
                    y1 = (y[0], y[1])
                    y2 = (y[2], 1)
                    x_train_images.append(np.asarray(img_seq))
                    x_train_points.append(x)
                    y_train_points.append(np.asarray(y1))
                    y_train_probs.append(np.asarray(y2))
                elif numpoints < len(point_seq) - 2:
                    x = np.asarray(point_seq[0:numpoints + 1])
                    y = point_seq[numpoints + 1]
                    y1 = (y[0], y[1])
                    y2 = (y[2], 0)
                    x_train_images.append(np.asarray(img_seq))
                    x_train_points.append(x)
                    y_train_points.append(np.asarray(y1))
                    y_train_probs.append(np.asarray(y2))
    x_train_images = np.asarray(x_train_images)
    x_train_points = np.asarray(x_train_points)
    y_train_points = np.asarray(y_train_points)
    y_train_probs = np.asarray(y_train_probs)
    return (x_train_images, x_train_points, y_train_points, y_train_probs)


## Can only run fit if all the sequence lengths are equal
## So we iterate through the possible combinations of image
## sequence lengths and pen points sequence lengths and train
## as seperate batches
def train():
    for numpoints in range(1, 1950):
        for numimgs in range(1, 60):
            print(numpoints)
            print(numimgs)
            (x_train_images, x_train_points, y_train_points, y_train_probs) = get_training_data(numpoints, numimgs)
            if x_train_images.shape[0] != 0:
                gen_model.fit([x_train_images, x_train_points], [y_train_points, y_train_probs])
        gen_model.save_weights('./checkpoints/checkpoint')



# gen_model.fit([x_train_images, x_train_points], [y_train_points, y_train_probs])

# shortdata max and mins
# max_point_seq_length = 1469
# max_img_seq_length = 55
# min_point_seq_length = 103
# min_img_seq_length = 4

#trainingdata max and mins
# max_point_seq_length = 1940
# max_img_seq_length = 58
# min_point_seq_length = 37
# min_img_seq_length = 1
