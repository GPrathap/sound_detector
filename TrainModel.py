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
import APIs as api
import tensorflow as tf


numOfClasses = 10
imagewidth = 16


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




clips_10 , datasetXForConvolution, datasetYForConvolution , datasetXForFull, datasetYForFull = api.load_dataset('ESC-10')
datasetXForConvolution,datasetYForConvolution = reconstructFeatureMatrix(datasetXForConvolution, datasetYForConvolution)

datasetXLengthForConvolution =len(datasetXForConvolution[0])
print datasetXLengthForConvolution
datasetYLengthForConvolution = len(datasetYForConvolution[0])

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