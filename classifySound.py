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



class ClassifiedSound:
    ################### add placeholder start################################################
    ####much simpler with STFT

    def __init__(self, soundChunk):
        self.soundChunk = soundChunk[0:40960]
        self.width = 80
        self.height = 40
        self.numOfClasses = 10
        self.modlePath = "neural_net/neural_net.ckpt"
        self.imagewidth = 80
        soundChunk = np.array(soundChunk)
        self.audoData = api.ClipRealTime(self.soundChunk)
        clips_10_test, datasetXForConvolution1, datasetYForConvolution1, datasetXForFull1, datasetYForFull1 = api.load_dataset_realtime(
            self.audoData)
        self.datasetXForConvolution_test, self.datasetYForConvolution_test = self.reconstructFeatureMatrixRealTime(
            datasetXForConvolution1,
            datasetYForConvolution1)
        self.datasetXLengthForConvolution = len(self.datasetXForConvolution_test[0])
        self.datasetYLengthForConvolution = len(self.datasetYForConvolution_test[0])
        self.xs = tf.placeholder(tf.float32, [None, self.datasetXLengthForConvolution], name='x_input')
        self.ys = tf.placeholder(tf.float32, [None, self.datasetYLengthForConvolution], name='y_input')
        self.keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(self.xs, [-1, self.width, self.height, 1])  ##is this needed based on the binning protocol
        #tf.image_summary('inputs', x_image, 1)
        ################### add layers start################################################

        ## conv1 layer with pooling##
        depth1 = 10
        W_conv1 = self.weight_variable([16, 16, 1, depth1])  # patch 5x5, in size 1, out size 32
        b_conv1 = self.bias_variable([depth1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1, [1, 1, 1, 1], 'SAME') + b_conv1)  # output size 28x28x32
        h_pool1 = self.max_pool_2x2(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')  # output size 14x14x32

        ## conv2 layer with pooling##
        depth2 = 10
        W_conv2 = self.weight_variable([10, 4, depth1, depth2])  # patch 5x5, in size 32, out size 64
        b_conv2 = self.bias_variable([depth2])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)  # output size 14x14x64
        h_pool2 = self.max_pool_2x2(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')  # output size 7x7x64

        ## func1 layer ##
        nodes1 = 20
        h_pool2_flat = tf.reshape(h_pool2, [-1, (self.width / 4) * (self.height / 4) * depth2])
        h_fc1 = self.add_layer(h_pool2_flat, (self.width / 4) * (self.height / 4) * depth2, nodes1, 3, tf.tanh)

        self.add_layer(h_fc1, 20, self.datasetYLengthForConvolution, 4)
        # pre_prediction = add_layer(h_fc1, nodes1, datasetYLengthForConvolution, 4)

        ##########################Session variables####################################
        # saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        ###########################################################
        self.new_saver = tf.train.import_meta_graph('/home/geesara/Documents/KVA/sound_detector/neural_net/my-model-0.meta')
        self.new_saver.restore(self.sess, '/home/geesara/Documents/KVA/sound_detector/neural_net/my-model-0')
        #train_op = tf.get_collection('train_op')[0]

        self.prediction1 = tf.get_collection('prediction')[0]


    def calculateClass(self, soundChunk):
        self.soundChunk = soundChunk
        self.audoData = api.ClipRealTime(self.soundChunk)
        clips_10_test, datasetXForConvolution1, datasetYForConvolution1, datasetXForFull1, datasetYForFull1 = api.load_dataset_realtime(
            self.audoData)
        self.datasetXForConvolution_test, self.datasetYForConvolution_test = self.reconstructFeatureMatrixRealTime(
            datasetXForConvolution1,
            datasetYForConvolution1)
        self.datasetXLengthForConvolution = len(self.datasetXForConvolution_test[0])
        self.datasetYLengthForConvolution = len(self.datasetYForConvolution_test[0])

        gh = []
        gh.append(self.datasetXForConvolution_test[0])
        y_pre1 = self.sess.run(self.prediction1, feed_dict={self.xs: gh, self.keep_prob: 1})
        # y_pre1 = sess.run(prediction, feed_dict={xs: datasetXForConvolution_test, keep_prob: 1})
        y_pre2 = tf.nn.softmax(y_pre1)
        print y_pre2
        correct_prediction = tf.equal(tf.argmax(y_pre2, 1), tf.argmax(gh, 1))
        self.sess.run(y_pre2, feed_dict={self.xs: gh})[0]
        # meta_graph_def = tf.train.export_meta_graph(filename='/home/geesara/Documents/KVA/sound_detector/neural_net/neural_net.ckpt.meta', as_text=True)
        classesList = -np.sort(-(self.sess.run(y_pre2, feed_dict={self.xs: gh})[0]))
        self.topGuesses = []
        for i in range(0, 3):
            self.topGuesses.append(classesList.tolist().index(classesList[i]))

        return self.topGuesses

    def __exit__(self, exception_type, exception_value, traceback):
        # if exception_type is not None:
        # print exception_type, exception_value, traceback
        del self.prediction1
        del self.audoData
        del self.sess
        del self.xs
        del self.ys
        del self.keep_prob

    def remove(self):
        del self.prediction1
        del self.audoData
        del self.sess
        del self.xs
        del self.ys
        del self.keep_prob
        del self.datasetXForConvolution_test
        del self.datasetYForConvolution_test

    def getClassifedClass(self):
        return self.topGuesses

    def reconstructFeatureMatrix(self, datasetXForConvolution, datasetYForConvolution):
        newXDataSetX = []
        newYDataSetY = []
        number_of_datasets = int(len(datasetXForConvolution))
        for i in range(0, (number_of_datasets - self.imagewidth)):
            if 2 not in (map(add, datasetYForConvolution[i], datasetYForConvolution[i + self.imagewidth])):
                pass
            else:
                newYDataSetY.append(datasetYForConvolution[i])
                temp = datasetXForConvolution[i:i + self.imagewidth]
                temp2 = [x for y in temp for x in y]
                newXDataSetX.append(temp2)
        return newXDataSetX, newYDataSetY

    def reconstructFeatureMatrixRealTime(self, datasetXForConvolution, datasetYForConvolution):
        newXDataSetX = []
        newYDataSetY = []
        number_of_datasets = 1
        for i in range(0, number_of_datasets):
            newYDataSetY.append(datasetYForConvolution[i])
            temp = datasetXForConvolution[i:i + self.imagewidth]
            temp2 = [x for y in temp for x in y]
            newXDataSetX.append(temp2)
        return newXDataSetX, newYDataSetY


    #####################convolutional layer creation tools start###################################################

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride, pad):
        '''eg stride=[1, 1, 1, 1], pad='SAME' layername='convolutional layer1' '''
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=stride, padding=pad)

    def max_pool_2x2(self, x, kernelsize, stride, pad):
        return tf.nn.max_pool(x, ksize=kernelsize, strides=stride, padding=pad)

    def add_layer(self, inputs, in_size, out_size, n_layer, activation_function=None):
        # add one more layer and return the output of this layer
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
                # here to dropout
                Wx_plus_b = tf.nn.dropout(Wx_plus_b, self.keep_prob)
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b, )
            return outputs
    # should be divisible by 4



