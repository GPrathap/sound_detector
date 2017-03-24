
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

slim = tf.contrib.slim

class CNNModel2:

    def __init__(self,project_dir, meta_info_data_set):

        meta_info_file = project_dir + "/neuralnet/net/cnn/model2/config.json"

        with open(meta_info_file) as data_file:
            meta_info = json.load(data_file)
            self.learning_rate = float(meta_info["net"]["learning_rate"])
            self.conv1_features = int(meta_info["net"]["conv1_features"])
            self.conv2_features = int(meta_info["net"]["conv2_features"])
            self.max_pool_size1 = int(meta_info["net"]["max_pool_size1"])
            self.max_pool_size2 = int(meta_info["net"]["max_pool_size2"])
            self.fully_connected_size1 = int(meta_info["net"]["fully_connected_size1"])
            self.filter_side = int(meta_info["net"]["filter_side"])
            self.dropout = int(meta_info["net"]["dropout"])
            self.training_iters = int(meta_info["net"]["training_iters"])
            self.display_step = int(meta_info["net"]["display_step"])
            self.strides_layer1 = int(meta_info["net"]["strides_layer1"])
            self.strides_layer2 = int(meta_info["net"]["strides_layer2"])
            self.model_path = project_dir + str(meta_info["net"]["model_path"])
            self.logs_path = project_dir + str(meta_info["net"]["logs_path"])
            self.keep_prob = float(meta_info["net"]["keep_prob"])

            self.num_channels = int(meta_info_data_set["num_channels"])
            self.n_classes = int(meta_info_data_set["number_of_class"])
            self.image_width = int(meta_info_data_set["generated_image_width"])
            self.image_height = int(meta_info_data_set["generated_image_height"])
            self.feature_vector_size = self.image_height*self.image_width

            with tf.name_scope('Inputs'):
                self.x_input = tf.placeholder(tf.float32, [None, self.feature_vector_size], name='InputData')
                self.y_target = tf.placeholder(tf.int32, [None, self.n_classes], name='LabelData')

            with tf.name_scope("Dropout"):
                self.keep_prob = tf.placeholder(tf.float32)

    def block35(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
      """Builds the 35x35 resnet block."""
      with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
          net = activation_fn(net)
      return net


    def block17(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
      """Builds the 17x17 resnet block."""
      with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                      scope='Conv2d_0b_1x7')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                      scope='Conv2d_0c_7x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
          net = activation_fn(net)
      return net


    def block8(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
      """Builds the 8x8 resnet block."""
      with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                      scope='Conv2d_0b_1x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                      scope='Conv2d_0c_3x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
          net = activation_fn(net)
      return net



    def conv_net(self, inputs,  is_training=True,
                            dropout_keep_prob=0.8,
                            reuse=None,
                            scope='InceptionResnetV2'):
      """Creates the Inception Resnet V2 model.


      Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.

      Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
      """
      end_points = {}

      inputs = tf.reshape(self.x_input, shape=[-1, self.image_width, self.image_height, self.num_channels])

      with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
          with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                              stride=1, padding='SAME'):

            # 149 x 149 x 32
            net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                              scope='Conv2d_1a_3x3')
            end_points['Conv2d_1a_3x3'] = net
            # 147 x 147 x 32
            net = slim.conv2d(net, 32, 3, padding='VALID',
                              scope='Conv2d_2a_3x3')
            end_points['Conv2d_2a_3x3'] = net
            # 147 x 147 x 64
            net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
            end_points['Conv2d_2b_3x3'] = net
            # 73 x 73 x 64
            net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                  scope='MaxPool_3a_3x3')
            end_points['MaxPool_3a_3x3'] = net
            # 73 x 73 x 80
            net = slim.conv2d(net, 80, 1, padding='VALID',
                              scope='Conv2d_3b_1x1')
            end_points['Conv2d_3b_1x1'] = net
            # 71 x 71 x 192
            net = slim.conv2d(net, 192, 3, padding='VALID',
                              scope='Conv2d_4a_3x3')
            end_points['Conv2d_4a_3x3'] = net
            # 35 x 35 x 192
            net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                  scope='MaxPool_5a_3x3')
            end_points['MaxPool_5a_3x3'] = net

            # 35 x 35 x 320
            with tf.variable_scope('Mixed_5b'):
              with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
              with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                            scope='Conv2d_0b_5x5')
              with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                            scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                            scope='Conv2d_0c_3x3')
              with tf.variable_scope('Branch_3'):
                tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                             scope='AvgPool_0a_3x3')
                tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                           scope='Conv2d_0b_1x1')
              net = tf.concat(axis=3, values=[tower_conv, tower_conv1_1,
                                  tower_conv2_2, tower_pool_1])

            end_points['Mixed_5b'] = net
            net = slim.repeat(net, 10, self.block35, scale=0.17)

            # 17 x 17 x 1088
            with tf.variable_scope('Mixed_6a'):
              with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                         scope='Conv2d_1a_3x3')
              with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                            scope='Conv2d_0b_3x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                            stride=2, padding='VALID',
                                            scope='Conv2d_1a_3x3')
              with tf.variable_scope('Branch_2'):
                tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                             scope='MaxPool_1a_3x3')
              net = tf.concat(axis=3, values=[tower_conv, tower_conv1_2, tower_pool])

            end_points['Mixed_6a'] = net
            net = slim.repeat(net, 20, self.block17, scale=0.10)

            # Auxiliary tower
            with tf.variable_scope('AuxLogits'):
              aux = slim.avg_pool2d(net, 5, stride=3, padding='VALID',
                                    scope='Conv2d_1a_3x3')
              aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
              aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                                padding='VALID', scope='Conv2d_2a_5x5')
              aux = slim.flatten(aux)
              aux = slim.fully_connected(aux, self.n_classes, activation_fn=None,
                                         scope='Logits')
              end_points['AuxLogits'] = aux

            with tf.variable_scope('Mixed_7a'):
              with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
              with tf.variable_scope('Branch_1'):
                tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                            padding='VALID', scope='Conv2d_1a_3x3')
              with tf.variable_scope('Branch_2'):
                tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                            scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                            padding='VALID', scope='Conv2d_1a_3x3')
              with tf.variable_scope('Branch_3'):
                tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                             scope='MaxPool_1a_3x3')
              net = tf.concat(axis=3, values=[tower_conv_1, tower_conv1_1,
                                  tower_conv2_2, tower_pool])

            end_points['Mixed_7a'] = net

            net = slim.repeat(net, 9, self.block8, scale=0.20)
            net = self.block8(net, activation_fn=None)

            net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
            end_points['Conv2d_7b_1x1'] = net

            with tf.variable_scope('Logits'):
              end_points['PrePool'] = net
              net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                    scope='AvgPool_1a_8x8')
              net = slim.flatten(net)

              net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='Dropout')

              end_points['PreLogitsFlatten'] = net
              logits = slim.fully_connected(net, self.n_classes, activation_fn=None,
                                            scope='Logits')
              end_points['Logits'] = logits
              end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
        return logits, end_points

    def inception_resnet_v2_arg_scope(self, weight_decay=0.00004,
                                      batch_norm_decay=0.9997,
                                      batch_norm_epsilon=0.001):
      """Yields the scope with the default parameters for inception_resnet_v2.

      Args:
        weight_decay: the weight decay for weights variables.
        batch_norm_decay: decay for the moving average of batch_norm momentums.
        batch_norm_epsilon: small float added to variance to avoid dividing by zero.

      Returns:
        a arg_scope with the parameters needed for inception_resnet_v2.
      """
      # Set weight_decay for weights in conv2d and fully_connected layers.
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          biases_regularizer=slim.l2_regularizer(weight_decay)):

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
          return scope
