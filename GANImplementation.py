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
#import tensorflow_gan as tfgan

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def create_gen_model():

    img_input = layers.Input(shape = (48, 48, 1), name = 'ImagesIn')

    conv1 = layers.Conv2D(10, 3, padding = 'same', activation='relu')(img_input)
    maxpool1 = layers.MaxPooling2D((2,2))(conv1)
    conv2 = layers.Conv2D(10, 3, padding = 'same', activation='relu')(maxpool1)
    maxpool2 = layers.MaxPooling2D((2,2))(conv2)
    conv3 = layers.Conv2D(10, 3, padding = 'same', activation='relu')(maxpool2)
    flatten = layers.Flatten()(conv3)

    ## pen_inputs: positions as offsets from previous point
    ## (posn_x, posn_y, pen_up)
    pen_input = layers.Input(shape = 3, name = 'PenInputs')
    concat = layers.concatenate([pen_input, flatten])
    reshape = layers.Reshape((1, 1443))(concat)

    lstm1, lstm1_state_h, lstm1_state_c = layers.LSTM(100, return_sequences=True, return_state=True)(reshape)
    lstm1_next_state = [lstm1_state_h, lstm1_state_c]
    lstm2, lstm2_state_h, lstm2_state_c = layers.LSTM(100, return_sequences=True, return_state=True)(lstm1)
    lstm2_next_state = [lstm2_state_h, lstm2_state_c]
    lstm3, lstm3_state_h, lstm3_state_c = layers.LSTM(100, return_state = True)(lstm2)
    lstm3_next_state = [lstm3_state_h, lstm3_state_c]

    ## Means and variances for position Gaussians
    mu = layers.Dense(2, activation='linear')(lstm3)
    sigma = layers.Dense(2, activation='relu')(lstm3)
    ## Probabilities for Bernoulli's to determine symbol completion and
    ## stroke endings
    p1 = layers.Dense(1, activation='sigmoid')(lstm3)
    p2 = layers.Dense(1, activation='sigmoid')(lstm3)


    return Model(inputs = [img_input, pen_input],
                outputs= [mu, sigma, p1, p2, lstm1_next_state, lstm2_next_state, lstm3_next_state])

def create_disc_model():
    ## The discriminator model will have the same structure as the generator
    ## but the outputs will only return the probability that the current
    ## symbol is finished being drawn, and
    img_input = layers.Input(shape = (48, 48, 1), name = 'ImagesIn')

    conv1 = layers.Conv2D(10, 3, padding = 'same', activation='relu')(img_input)
    maxpool1 = layers.MaxPooling2D((2,2))(conv1)
    conv2 = layers.Conv2D(10, 3, padding = 'same', activation='relu')(maxpool1)
    maxpool2 = layers.MaxPooling2D((2,2))(conv2)
    conv3 = layers.Conv2D(10, 3, padding = 'same', activation='relu')(maxpool2)
    flatten = layers.Flatten()(conv3)

    pen_input = layers.Input(shape = 3, name = 'PenInputs')
    concat = layers.concatenate([pen_input, flatten])
    reshape = layers.Reshape((1, 1443))(concat)

    lstm1, lstm1_state_h, lstm1_state_c = layers.LSTM(100, return_sequences=True, return_state=True)(reshape)
    lstm1_next_state = [lstm1_state_h, lstm1_state_c]
    lstm2, lstm2_state_h, lstm2_state_c = layers.LSTM(100, return_sequences=True, return_state=True)(lstm1)
    lstm2_next_state = [lstm2_state_h, lstm2_state_c]
    lstm3, lstm3_state_h, lstm3_state_c = layers.LSTM(100, return_state = True)(lstm2)
    lstm3_next_state = [lstm3_state_h, lstm3_state_c]

    ## return probability to look at next symbol in the sequence
    ## and probability (so far) that the input data is fake
    p = layers.Dense(2, activation='sigmoid')(lstm3)

    return Model(inputs = [img_input, pen_input],
                 outputs = [p, lstm1_next_state, lstm2_next_state, lstm3_next_state])

## This model was intended to be used to train the Generator.
## We would have initialized the weights before training using
## the gen_model weights, and disc_model weights to
def create_combined_model():

    img_input = layers.Input(shape = (48, 48, 1), name = 'ImagesIn')
    pen_input = layers.Input(shape = 3, name = 'PenInputs')

    gen_conv1 = layers.Conv2D(10, 3, padding = 'same', activation='relu', name = 'gen_conv1')(img_input)
    gen_maxpool1 = layers.MaxPooling2D((2,2), name = 'gen_maxpool1')(gen_conv1)
    gen_conv2 = layers.Conv2D(10, 3, padding = 'same', activation='relu', name = 'gen_conv2')(gen_maxpool1)
    gen_maxpool2 = layers.MaxPooling2D((2,2), name = 'gen_maxpool2')(gen_conv2)
    gen_conv3 = layers.Conv2D(10, 3, padding = 'same', activation='relu', name = 'gen_conv3')(gen_maxpool2)
    gen_flatten = layers.Flatten(name = 'gen_flatten')(gen_conv3)


    gen_concat = layers.concatenate([pen_input, gen_flatten])
    gen_reshape = layers.Reshape((1, 1443))(gen_concat)

    gen_lstm1, gen_lstm1_state_h, gen_lstm1_state_c = layers.LSTM(100, return_sequences=True, return_state=True, name = 'gen_lstm1')(gen_reshape)
    gen_lstm1_next_state = [gen_lstm1_state_h, gen_lstm1_state_c]
    gen_lstm2, gen_lstm2_state_h, gen_lstm2_state_c = layers.LSTM(100, return_sequences=True, return_state=True, name = 'gen_lstm2')(gen_lstm1)
    gen_lstm2_next_state = [gen_lstm2_state_h, gen_lstm2_state_c]
    gen_lstm3, gen_lstm3_state_h, gen_lstm3_state_c = layers.LSTM(100, return_state = True, name = 'gen_lstm3')(gen_lstm2)
    gen_lstm3_next_state = [gen_lstm3_state_h, gen_lstm3_state_c]

    mu = layers.Dense(2, activation='linear')(gen_lstm3)
    p1 = layers.Dense(1, activation='sigmoid')(gen_lstm3)

    disc_conv1 = layers.Conv2D(10, 3, padding = 'same', activation='relu', name = 'disc_conv1', trainable=False)(img_input)
    disc_maxpool1 = layers.MaxPooling2D((2,2), name = 'disc_maxpool1', trainable=False)(disc_conv1)
    disc_conv2 = layers.Conv2D(10, 3, padding = 'same', activation='relu', name = 'disc_conv2', trainable=False)(disc_maxpool1)
    disc_maxpool2 = layers.MaxPooling2D((2,2), name = 'disc_maxpool2', trainable=False)(disc_conv2)
    disc_conv3 = layers.Conv2D(10, 3, padding = 'same', activation='relu', name = 'disc_conv3', trainable=False)(disc_maxpool2)
    disc_flatten = layers.Flatten(name = 'disc_flatten')(disc_conv3)

    disc_concat = layers.concatenate([mu, p1, disc_flatten])
    disc_reshape = layers.Reshape((1, 1443))(disc_concat)

    disc_lstm1, disc_lstm1_state_h, disc_lstm1_state_c = layers.LSTM(100, return_sequences=True, return_state=True, name = 'disc_lstm1', trainable=False)(disc_reshape)
    disc_lstm1_next_state = [disc_lstm1_state_h, disc_lstm1_state_c]
    disc_lstm2, disc_lstm2_state_h, disc_lstm2_state_c = layers.LSTM(100, return_sequences=True, return_state=True, name = 'disc_lstm2', trainable=False)(disc_lstm1)
    disc_lstm2_next_state = [disc_lstm2_state_h, disc_lstm2_state_c]
    disc_lstm3, disc_lstm3_state_h, disc_lstm3_state_c = layers.LSTM(100, return_state = True, name = 'disc_lstm3', trainable=False)(disc_lstm2)
    disc_lstm3_next_state = [disc_lstm3_state_h, disc_lstm3_state_c]

    disc_p = layers.Dense(2, activation='sigmoid', trainable=False)(disc_lstm3)
    return Model(inputs = [img_input, pen_input],
                 outputs = [disc_p, gen_lstm1_next_state, gen_lstm2_next_state, gen_lstm3_next_state,
                               disc_lstm1_next_state, disc_lstm2_next_state, disc_lstm3_next_state])






def generate_prediction(disc_model, Image_seq, pen_seq):
    preds = []
    inputs = []
    lstm_size = 100
    lstm1_state = [None, None]
    lstm2_state = [None, None]
    lstm3_state = [None, None]
    image_index = 0
    counter = 0
    for point in pen_seq:
        print(counter)
        counter = counter + 1
        if image_index >= len(Image_seq):
            ## if the discriminator goes throughthe images too quickly, present blank images
            image = np.zeros_like(image)
            new_word = 0
        else:
            image_tuple = Image_seq[image_index]
            image = image_tuple[0].astype('float32')
            new_word = image_tuple[1]
        next_symbol = 0
        image_reshaped = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        pen_input = np.array((point[0], point[1], point[2], new_word)).astype('float32')
        pen_input_reshaped = np.reshape(pen_input, (1, pen_input.shape[0]))
        disc_model.layers[10].state = lstm1_state
        disc_model.layers[11].state = lstm2_state
        disc_model.layers[12].state = lstm3_state
        inputs.append([image_reshaped, pen_input, reshaped])
        (p, lstm1_next_state, lstm2_next_state, lstm3_next_state) = disc_model([image_reshaped, pen_input_reshaped])
        p = p.numpy()[0]
        preds.append(p[1])
        if p[0] > 1/2:
            image_index = image_index + 1
    return preds



## This function was generating synthetic data at one point,
## not sure how it broke, but the idea was to create an input
## sequence and input it into the combined model with training
## label = 1. Thus the combined model would be training the
## weights of the generator,
def generate_hw(gen_model, Image_seq):
    inkpoints = [(0, 0, 0)]
    point_limit = 100
    pen_input = np.zeros(3)
    lstm_size = 100
    lstm1_state = [None, None]
    lstm2_state = [None, None]
    lstm3_state = [None, None]
    inputs = []

    for image in Image_seq:
        image = image.astype('float32')
        points = 0
        next_symbol = 0
        image_reshaped = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        while points < point_limit and next_symbol == 0:

            pen_input_reshaped = np.reshape(pen_input, (1, pen_input.shape[0]))
            gen_model.layers[10].state = lstm1_state
            gen_model.layers[11].state = lstm2_state
            gen_model.layers[12].state = lstm3_state
            inputs.append([image_reshaped, pen_input_reshaped].copy())
            (mu, sigma, p1, p2, lstm1_state, lstm2_state, lstm3_state) = gen_model([image_reshaped, pen_input_reshaped], training = True)
            mu = mu.numpy()[0]
            sigma = sigma.numpy()[0]

            posn_x = sigma[0] * np.random.randn() + mu[0]
            posn_y = sigma[1] * np.random.randn() + mu[1]
            pen_up = np.random.binomial(1, p1[0][0])
            next_symbol = np.random.binomial(1, p2[0][0])
            # prev_point = inkpoints[-1]
            inkpoints.append((posn_x, posn_y, pen_up))
            pen_input = np.array((posn_x, posn_y, pen_up))
            points = points + 1
            print(points, next_symbol)

    return (inkpoints, inputs)


def gen_train(img_seq, pen_seq):


    pass



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

def draw_absolutes(inkpoints):
    start = [inkpoints[0][0], inkpoints[0][1], inkpoints[0][2]]
    for i in range(len(inkpoints) - 1):
        next = inkpoints[i + 1]

        if start[2] == 0:
            color = 'black'
        else:
            color = 'white'
        plt.plot([start[0], next[0]], [-start[1], -next[1]], color=color)
        start[0] = next[0]
        start[1] = next[1]
        start[2] = next[2]

    plt.show()
# randomimg = np.random.randn(48, 48)
# randomimg = np.reshape(randomimg, (1, randomimg.shape[0], randomimg.shape[1], 1))
#
# randompen = np.random.randn(4)
# randompen = np.reshape(randompen, (1, randompen.shape[0]))
#
# output = model([randomimg, randompen], training=False)

## Helper funtion to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(outputs, targets):
    return cross_entropy(targets, outputs)

def generator_loss(outputs, targets):
    return cross_entropy(targets, outputs)

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)


# def train_step(gen_model, disc_model, imgSeq, writtenSeq):
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         gen_tape.watch(gen_model.trainable_variables)
#         disc_tape.watch(disc_model.trainable_variables)
#
#         gen_seq = generate_hw(gen_model, imgSeq)
#
#         real_output = generate_prediction(disc_model, imgSeq, writtenSeq)
#         fake_output = generate_prediction(disc_model, imgSeq, gen_seq)
#
#         gen_loss = generator_loss(fake_output, tf.ones_like(fake_output))
#         disc_loss = discriminator_loss(tf.concat([real_output, fake_output], 0),
#                               tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0))
#     # gradients_of_generator = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
#     # gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
#     #
#     # gen_optimizer.apply_gradients(zip(gradients_of_generator, gen_model.trainable_variables))
#     # disc_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_model.trainable_variables))
#     return (gen_seq, gen_loss, disc_loss, gen_tape, disc_tape)


(imgSeqs, WL) = pXML.xmltosequencepair(pXML.fn)
imgSeq = imgSeqs[0]
WL0 = WL[0]

gen_model = create_gen_model()

utils.plot_model(gen_model, 'gen_model.png', show_shapes=True)

gen_model.compile(optimizer=gen_optimizer,
                  loss=generator_loss)

comb_model = create_combined_model()

disc_model = create_disc_model()

utils.plot_model(disc_model, 'disc_model.png', show_shapes=True)

disc_model.compile(optimizer=disc_optimizer,
                loss=discriminator_loss)

comb_model = create_combined_model()

utils.plot_model(comb_model, 'comb_model.png', show_shapes=True)
comb_model.compile(optimizer= gen_optimizer, loss= generator_loss)



##(gen_seq, gen_loss, disc_loss, gen_tape, disc_tape) = train_step(gen_model, disc_model, imgSeq, WL0)

# preds1 = generate_prediction(disc_model, imgSeq, WL0)
#
# (gen_seq, gen_tape) = generate_hw(gen_model, imgSeq)
# fake_output = generate_prediction(disc_model, imgSeq, gen_seq)
# gen_loss = generator_loss(fake_output, tf.ones_like(fake_output))
# preds2 = generate_prediction(disc_model, imgSeq, inkpoints)

# draw_offsets(inkpoints)
