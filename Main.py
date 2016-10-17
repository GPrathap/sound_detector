import numpy as np
import pydub
import librosa
import os
import shutil
import urllib
import zipfile
import glob
import matplotlib
import matplotlib.pyplot as plt
from operator import add

from sys import getsizeof

import tensorflow as tf
import numpy as np
from operator import add

import seaborn as sb
import time

sb.set(style="white", palette="muted")

import pandas as pd
import random
random.seed(20150420)
numOfClasses = 10

class Clip:
    """A single 5-sec long recording."""

    RATE = 44100  # All recordings in ESC are 44.1 kHz
    FRAME = 1024  # Frame size in samples

    class Audio:
        """The actual audio data of the clip.

            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage.
        """

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            # Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
            self.data = pydub.AudioSegment.silent(duration=5000)
            self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path)[0:5000])
            self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float
            return (self)

        def __exit__(self, exception_type, exception_value, traceback):
            #if exception_type is not None:
                #print exception_type, exception_value, traceback
            del self.data
            del self.raw

    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)
        self.directory = os.path.dirname(self.path)
        self.category = self.directory.split('/')[-1]

        self.audio = Clip.Audio(self.path)

        with self.audio as audio:
            self._compute_mfcc(audio)
            self._compute_zcr(audio)
            self._compute_fft(audio)

    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length, 128 bands)
        self.melspectrogram = librosa.feature.melspectrogram(audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME)
        self.logamplitude = librosa.logamplitude(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=256).transpose()

    def _compute_fft(self, audio):
        self.fft = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))

        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            ps = np.fft.fft(frame)
            self.fft.append(np.abs(ps))

        self.fft = np.asarray(self.fft)



    def _compute_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))

        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(frame)))))

        self.zcr = np.asarray(self.zcr)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME):(index + 1) * Clip.FRAME]

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)


def getClassArray():
    return [0] * numOfClasses


def load_dataset(name):
    """Load all dataset recordings into a nested list."""
    clips = []

    datasetXForConvolution = []
    datasetYForConvolution = []
    datasetXForFull = []
    datasetYForFull = []
    for directory in sorted(os.listdir('{0}/'.format(name))):
        directory = '{0}/{1}'.format(name, directory)
        if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
            print('Parsing ' + directory)
            category = []
            for clip in sorted(os.listdir(directory)):
                if clip[-3:] == 'ogg':
                    audioFile = Clip('{0}/{1}'.format(directory, clip))
                    numberOfWindows = len(audioFile.mfcc)

                    featureSetPerWindowXForConvolution = audioFile.mfcc
                    datasetXForConvolution.append(featureSetPerWindowXForConvolution.tolist())

                    featureSetPerWindowXForFull = audioFile.zcr
                    datasetXForFull.append(featureSetPerWindowXForFull.tolist())

                    for i in range(0, numberOfWindows):
                        classes = getClassArray()
                        classes[int('{1}'.format(name, directory).split("/")[1].split("-")[0]) - 1] = 1
                        datasetYForConvolution.append(classes)
                        ##after adding new features add it to either one
                        datasetYForFull.append(classes)

                    category.append(audioFile)
            clips.append(category)

    print('All {0} recordings loaded.'.format(name))
    return clips , datasetXForConvolution[0], datasetYForConvolution , datasetXForFull, datasetYForFull

def reconstructFeatureMatrix(datasetXForConvolution,datasetYForConvolution):
    newXDataSetX = []
    newYDataSetY = []
    number_of_datasets = int(len(datasetXForConvolution))
    for i in range(0, (number_of_datasets - imagewidth)):
        if 2 not in (map(add, datasetYForConvolution[i], datasetYForConvolution[i + imagewidth])):
            pass
        else:
            newYDataSetY.append(datasetYForConvolution[i])
            temp = datasetXForConvolution[i:i + imagewidth]
            temp2 = [x for y in temp for x in y]
            newXDataSetX.append(temp2)
    return newXDataSetX, newYDataSetY

def add_subplot_axes(ax, position):
    box = ax.get_position()
    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]
    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], axisbg='w')


def plot_clip_overview(clip, ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])

    with clip.audio as audio:
        ax_waveform.plot(np.arange(0, len(audio.raw)) / float(Clip.RATE), audio.raw)
        ax_waveform.get_xaxis().set_visible(False)
        ax_waveform.get_yaxis().set_visible(False)
        ax_waveform.set_title('{0} \n {1}'.format(clip.category, clip.filename), {'fontsize': 8}, y=1.03)

        librosa.display.specshow(clip.logamplitude, sr=Clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
        ax_spectrogram.get_xaxis().set_visible(False)
        ax_spectrogram.get_yaxis().set_visible(False)


def plot_single_clip(clip):
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clip.mfcc)[1]))
    MFCC = pd.DataFrame(clip.mfcc[:, :], columns=col_names)

    f = plt.figure(figsize=(10, 6))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.75])
    ax_mfcc.set_xlim(-400, 400)
    ax_zcr = add_subplot_axes(ax, [0.0, 0.85, 1.0, 0.05])
    ax_zcr.set_xlim(0.0, 1.0)

    plt.title('Feature distribution across frames of a single clip ({0} : {1})'.format(clip.category, clip.filename), y=1.5)
    sb.boxplot(data=MFCC, orient='h', order=list(reversed(MFCC.columns)), ax=ax_mfcc)
    sb.boxplot(data=pd.DataFrame(clip.zcr, columns=['ZCR']), orient='h', ax=ax_zcr)
    plt.show()



def plot_single_feature_one_clip(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.10)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[2], ax=ax)


def plot_single_feature_all_clips(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.boxplot(feature, vert=False, orient='h', order=list(reversed(feature.columns)), ax=ax)


def plot_single_feature_aggregate(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[1], ax=ax)


def generate_feature_summary(dataset, category, clip, coefficient):
    title = "{0} : {1}".format(dataset[category][clip].category, dataset[category][clip].filename)
    MFCC = pd.DataFrame()
    aggregate = []
    for i in range(0, len(dataset[category])):
        MFCC[i] = dataset[category][i].mfcc[:, coefficient]
        aggregate = np.concatenate([aggregate, dataset[category][i].mfcc[:, coefficient]])

    f = plt.figure(figsize=(14, 12))
    f.subplots_adjust(hspace=0.6, wspace=0.3)

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

    ax1.set_xlim(0.0, 0.5)
    ax2.set_xlim(-100, 250)
    ax4.set_xlim(-100, 250)

    plot_single_feature_one_clip(dataset[category][clip].zcr, 'ZCR distribution across frames\n{0}'.format(title), ax1)
    plot_single_feature_one_clip(dataset[category][clip].mfcc[:, coefficient],
                                 'MFCC_{0} distribution across frames\n{1}'.format(coefficient, title), ax2)

    plot_single_feature_all_clips(MFCC, 'Differences in MFCC_{0} distribution\nbetween clips of {1}'.format(coefficient,
                                                                                    dataset[
                                                                                                                category][
                                                                                                                clip].category),
                                  ax3)

    plot_single_feature_aggregate(aggregate,'Aggregate MFCC_{0} distribution\n(bag-of-frames across all clips\nof {1})'.format(
                                      coefficient, dataset[category][clip].category), ax4)
    plt.show()



# all_recordings = glob.glob('ESC-50/*/*.ogg')
# clip = Clip(all_recordings[random.randint(0, len(all_recordings) - 1)])
#
# with clip.audio as audio:
#     plt.subplot(2, 1, 1)
#     plt.title('{0} : {1}'.format(clip.category, clip.filename))
#     plt.plot(np.arange(0, len(audio.raw)) / 44100.0, audio.raw)
#     plt.subplot(2, 1, 2)
#     librosa.display.specshow(clip.logamplitude, sr=44100, x_axis='frames', y_axis='linear', cmap='RdBu_r')
#     print("-------------------------------------")

numOfClasses = 10
imagewidth = 16
clips_10 , datasetXForConvolution, datasetYForConvolution , datasetXForFull, datasetYForFull = load_dataset('ESC-10')
datasetXForConvolution,datasetYForConvolution = reconstructFeatureMatrix(datasetXForConvolution,datasetYForConvolution)

datasetXLengthForConvolution =len(datasetXForConvolution[0])
print datasetXLengthForConvolution
datasetYLengthForConvolution = len(datasetYForConvolution[0])

################### add layer modified for tensorboard start ################################################

###remove this part start#########

# def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
# # add one more layer and return the output of this layer
# layer_name = 'layer%s' % n_layer
# with tf.name_scope(layer_name):
# with tf.name_scope('weights'):
# Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
# tf.histogram_summary(layer_name + '/weights', Weights)
# with tf.name_scope('biases'):
# biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
# tf.histogram_summary(layer_name + '/biases', biases)
# with tf.name_scope('Wx_plus_b'):
# Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
# if activation_function is None:
# outputs = Wx_plus_b
# else:
# outputs = activation_function(Wx_plus_b, )
# tf.histogram_summary(layer_name + '/outputs', outputs)
# return outputs
###remove this part end#########

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.histogram_summary(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.histogram_summary(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            # here to dropout
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
            #outputs = activation_function(Wx_plus_b, layer_name , name=None)
        tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs


################### add layer modified for tensorboard end################################################


################### add Link layer start################################################
def add_link_layer(input_set_1, input_set_2, in_size_1, in_size_2, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    input_set_1_changed = tf.reshape(input_set_1, [-1, in_size_1])
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):		
            Weights = tf.Variable(tf.random_normal([in_size_1+in_size_2, out_size]), name='W')
            tf.histogram_summary(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.histogram_summary(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(tf.concat(1, [input_set_1_changed, input_set_2]), Weights), biases)
            # here to dropout
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
            #outputs = activation_function(Wx_plus_b, layer_name , name=None)
        tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs

# inputs_set_1 is the convolutional layer final output and its size should be the scalar size(total number of nodes in it)
# add one more layer and return the output of this layer


#####################convolutional layer creation tools start###################################################

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride, pad):
    '''eg stride=[1, 1, 1, 1], pad='SAME' layername='convolutional layer1' '''
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)


def max_pool_2x2(x, kernelsize, stride, pad):
    return tf.nn.max_pool(x, ksize=kernelsize, strides=stride, padding=pad)


# ''' kernelsize=[1,2,2,1], stride=[1,2,2,1], pad='SAME' '''
# stride [1, x_movement, y_movement, 1]


#####################convolutional layer creation tools end###################################################

#####################classification accuracy computer start###################################################

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


#####################classification accuracy computer end###################################################

################### add placeholder start################################################
####much simpler with STFT
width = 256
height = 16
# should be divisible by 4

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, datasetXLengthForConvolution], name='x_input')
    ys = tf.placeholder(tf.float32, [None, datasetYLengthForConvolution], name='y_input')
    keep_prob = tf.placeholder(tf.float32)


with tf.name_scope('input_reshape'):
    x_image = tf.reshape(xs, [-1, width, height, 1])  ##is this needed based on the binning protocol
    tf.image_summary('inputs', x_image, 1)


################### add placeholder end################################################

#########################################################################################################
#########################################################################################################
#########################################################################################################

################### add layers start################################################

## conv1 layer with pooling##
depth1 = 10
W_conv1 = weight_variable([16, 16, 1, depth1])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([depth1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, [1, 1, 1, 1], 'SAME') + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')  # output size 14x14x32

## conv2 layer with pooling##
depth2 = 10
W_conv2 = weight_variable([10, 4, depth1, depth2])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([depth2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')  # output size 7x7x64

##fully connected layer 1
fully_connected_layer_1 = add_layer(xs,datasetXLengthForConvolution, 10, 1, tf.nn.relu)

fully_connected_layer_2 = add_layer(fully_connected_layer_1, 10, 10, 2, tf.nn.relu)

link_layer = add_link_layer(h_pool2, fully_connected_layer_2, (width / 4) * (height / 4) * depth2, 10, 10, 3, tf.nn.relu)


## func1 layer ##
#nodes1 = 20
#h_pool2_flat = tf.reshape(h_pool2, [-1, (width / 4) * (height / 4) * depth2])
# h_fc1 = add_layer(h_pool2_flat, (width / 4) * (height / 4) * depth2, nodes1, 3, tf.tanh)



# W_fc1 = weight_variable([(width/4)*(height/4)*depth2, nodes1])
# b_fc1 = bias_variable([nodes1])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

## func2 layer ##
# W_fc2 = weight_variable([nodes1, nodes2])
# b_fc2 = bias_variable([nodes2])
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

pre_prediction = add_layer(link_layer, 10, datasetYLengthForConvolution, 4)
# pre_prediction = add_layer(h_fc1, nodes1, datasetYLengthForConvolution, 4)


# uses softmax cost function and the adams optimizer trainer


#########################################################################################################
# temp1=100  #nodes per layer for now
# # add hidden layer1
# l1 = add_layer(xs, datasetXLengthForConvolution, temp1, n_layer=1, activation_function=tf.tanh)

# # add hidden layer2
# l2 = add_layer(l1, temp1, temp1, n_layer=2, activation_function=tf.tanh)

# # add output layer
# prediction = add_layer(l2, temp1, datasetYLengthForConvolution, n_layer=3, activation_function=tf.sigmoid)
#########################################################################################################

################### add layers end#######################################################################


################### add cost function and optimizer start################################################


#prediction = tf.nn.softmax(pre_prediction)
#tf.histogram_summary('prediction', prediction)


#total_loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), [1]))
##  this is equivalent to
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pre_prediction, ys))



# the error between prediction and real data
#with tf.name_scope('loss'):
    # total loss
    #loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) + (1 - ys) * tf.log(1 - prediction),
        #reduction_indices=[1]))
    # least squares
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
    # reduction_indices=[1]))
    # cross_entropy
    #loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction + 1e-50),reduction_indices=[1]))

#tf.scalar_summary('loss', loss)

with tf.name_scope('train'):
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
################### add cost function and optimizer end################################################



##########################Session variables####################################
sess = tf.Session()
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter("logs/train", sess.graph)
# test_writer = tf.train.SummaryWriter("logs/test", sess.graph)
sess.run(tf.initialize_all_variables())
###########################################################

for i in range(20):
    sess.run(train_step, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 0.5})
    if i % 2 == 0:
        train_result = sess.run(merged, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 1})
         # test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        loss_value = sess.run(loss, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 1})
        print(loss_value)

# previous_loss_value = 20
# epsilon = 0.0001
# loss_value = 10
# counter = 0
# i = 0
# while (counter < 5):
#     sess.run(train_step, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 0.5})
#     if i % 5 == 0:
#         train_result = sess.run(merged, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 1})
#         # test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
#         train_writer.add_summary(train_result, i)
#         # test_writer.add_summary(test_result, i)
#         # print(compute_accuracy(
#         # mnist.test.images, mnist.test.labels))
#     previous_loss_value = loss_value
#     loss_value = sess.run(loss, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 1})
#
#     print("{:.9f}".format(loss_value))
#     print("{:.9f}".format(loss_value - previous_loss_value))
#     if ((previous_loss_value - loss_value) < epsilon):
#         counter += 1
#
#     else:
#         counter = 0
#
#     i = i + 1
##############################################


##############################################



# for i in range(1000):
# sess.run(train_step, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution, keep_prob: 0.5})
# if i % 50 == 0:
# result = sess.run(merged,
# feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution})
# writer.add_summary(result, i)
# loss_value = sess.run(loss, feed_dict={xs: datasetXForConvolution, ys: datasetYForConvolution})
# print(loss_value)




# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs


































