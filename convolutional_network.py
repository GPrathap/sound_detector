# Introductory CNN Model: MNIST Digits
# ---------------------------------------
#
# In this example, we will download the MNIST handwritten
# digits and create a simple CNN network to predict the
# digit category (0-9)

import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import NotFoundError

from loader import DataLoader

ops.reset_default_graph()


# Set model parameters
batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = 20
image_height = 20
num_channels = 1  # greyscale = 1 channel
generations = 500
eval_every = 5
n_classes = 10
conv1_features = 32
conv2_features = 64
max_pool_size1 = 2  # NxN window for 1st max pool layer
max_pool_size2 = 2  # NxN window for 2nd max pool layer
fully_connected_size1 = 100
filter_side = 5
dropout = 0.75 # Dropout, probability to keep units
keep_prob = tf.placeholder(tf.float32) #dropout
training_iters = 200000
display_step = 10
n_input = 400 # MNIST data input (img shape: 28*28)

with tf.name_scope('Weights'):
# Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([filter_side, filter_side, num_channels, conv1_features]), name='Weights_wc1'),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([filter_side, filter_side, conv1_features, conv2_features]), name='Weights_wc2'),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([5*5*64, 1024]), name='Weights_wd1'),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd2': tf.Variable(tf.random_normal([1024, 1024]), name='Weights_wd2'),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]), name='Weights_out')
    }

with tf.name_scope('Biases'):

    biases = {
        'bc1': tf.Variable(tf.random_normal([conv1_features]), name='bc1'),
        'bc2': tf.Variable(tf.random_normal([conv2_features]), name='bc2'),
        'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
        'bd2': tf.Variable(tf.random_normal([1024]), name='bd2'),
        'out': tf.Variable(tf.random_normal([n_classes]), name='out')
    }

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return (100. * num_correct / batch_predictions.shape[0])

# Declare model placeholders
# x_input_shape = (batch_size, image_width, image_height, num_channels)
with tf.name_scope('Inputs'):
    x_input = tf.placeholder(tf.float32, [None, n_input], name='InputData')
    y_target = tf.placeholder(tf.int32, [None, n_classes], name='LabelData')
# eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
# eval_input = tf.placeholder(tf.float32, [None, n_input])
# eval_target = tf.placeholder(tf.int32, [None, n_classes])

# Initialize Model Operations
def conv_net(input_data, weights, biases, keep_prob):

    input_data = tf.reshape(input_data, shape=[-1, image_width, image_height, 1])

    # First Conv-ReLU-MaxPool Layer
    result_of_first_cnn_layer = conv2d(input_data, weights["wc1"], biases["bc1"], strides=1)
    tf.summary.histogram("result_of_first_cnn_layer", result_of_first_cnn_layer)
    result_of_first_max_polling_layer = maxpool2d(result_of_first_cnn_layer, k=2)

    # Second Conv-ReLU-MaxPool Layer
    result_of_second_cnn_layer = conv2d(result_of_first_max_polling_layer, weights["wc2"], biases["bc2"], strides=1)
    tf.summary.histogram("result_of_second_cnn_layer", result_of_second_cnn_layer)
    result_of_second_max_polling_layer = maxpool2d(result_of_second_cnn_layer, k=2)


    # Transform Output into a 1xN layer for next fully connected layer
    final_conv_shape = result_of_second_max_polling_layer.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(result_of_second_max_polling_layer, [-1 , final_shape])

    # First Fully Connected Layer
    fully_connected1 = tf.add(tf.matmul(flat_output, weights["wd1"]), biases["bd1"])
    fully_connected1 = tf.nn.relu(fully_connected1)
    tf.summary.histogram("fully_connected1", fully_connected1)

    # Apply Dropout
    fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob)

    # Second Fully Connected Layer
    fully_connected2 = tf.add(tf.matmul(fully_connected1, weights["wd2"]), biases["bd2"])
    fully_connected2 = tf.nn.relu(fully_connected2)
    tf.summary.histogram("fully_connected2", fully_connected2)

    # Output, class prediction
    out = tf.add(tf.matmul(fully_connected2, weights['out']), biases['out'])

    return (out)

with tf.name_scope('Model'):
    model_predicted_output = conv_net(x_input, weights, biases, keep_prob)

# test_model_output = conv_net(eval_input, weights, biases, keep_prob)
# test_prediction = tf.nn.softmax(test_model_output)

# Declare Loss Function (softmax cross entropy)
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_predicted_output, labels=y_target))

# Define loss and optimizer
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(model_predicted_output, 1), tf.argmax(y_target, 1))

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


loader = DataLoader('TRAIN-10', "audio_clips_segmentation.tfrecords", 512, 1,1, 2)
model_path = "/home/runge/projects/sound_detector/neuralnet/model/model.ckpt"
logs_path = '/home/runge/projects/sound_detector/neuralnet/graph/'

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
# loader.create_one_big_file("ogg")
# Launch the graph
with tf.Session() as sess:
    image, label = loader.inputs()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    try:
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
    except NotFoundError:
        print("There is a no model which has been saved previously in this directory: %s" % model_path)

    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Keep training until reach max iterations
    try:
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            # Run training steps or whatever
            batch_x, batch_y = sess.run([image, label])
            batch_x = np.transpose(batch_x[0:1] [0][0:1] [0][0:400])
            batch_y = np.transpose(batch_y[0:1] [0][0:1] [0][0:10])
            # Run optimization op (backprop)
            _, summary = sess.run([optimizer, merged_summary_op],  feed_dict={x_input: batch_x, y_target: batch_y,
                                           keep_prob: dropout})
            summary_writer.add_summary(summary, step * loader.batch_size + step)

            loss, acc = sess.run([cost, accuracy], feed_dict={x_input: batch_x,
                                                                  y_target: batch_y,
                                                                  keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= {:.6f}".format(
                loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            # TODO save the model as you required...
            saver.save(sess, model_path, global_step=step)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (loader.num_epochs, loader.batch_size))
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    sess.close()

    print("Optimization Finished!")
    print("Run the command line:\n" \
          "--> tensorboard --logdir=%s " \
          "\nThen open http://0.0.0.0:6006/ into your web browser" % logs_path)

    # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", \
    #       sess.run(accuracy, feed_dict={x_input: mnist.test.images[:256],
    #                                     y_target: mnist.test.labels[:256],
    #                                     keep_prob: 1.}))

    # Save model weights to disk


# test_acc = []
# for i in range(generations):
#     rand_index = np.random.choice(len(train_xdata), size=batch_size)
#     rand_x = train_xdata[rand_index]
#     rand_x = np.expand_dims(rand_x, 3)
#     rand_y = train_labels[rand_index]
#     train_dict = {x_input: rand_x, y_target: rand_y}
#
#     sess.run(train_step, feed_dict=train_dict)
#     temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
#     temp_train_acc = get_accuracy(temp_train_preds, rand_y)
#
#     if (i + 1) % eval_every == 0:
#         eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
#         eval_x = test_xdata[eval_index]
#         eval_x = np.expand_dims(eval_x, 3)
#         eval_y = test_labels[eval_index]
#         test_dict = {eval_input: eval_x, eval_target: eval_y}
#         test_preds = sess.run(test_prediction, feed_dict=test_dict)
#         temp_test_acc = get_accuracy(test_preds, eval_y)
#
#         # Record and print results
#         train_loss.append(temp_train_loss)
#         train_acc.append(temp_train_acc)
#         test_acc.append(temp_test_acc)
#         acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
#         acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
#         print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
#
# # Matlotlib code to plot the loss and accuracies
# eval_indices = range(0, generations, eval_every)
# # Plot loss over time
# plt.plot(eval_indices, train_loss, 'k-')
# plt.title('Softmax Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Softmax Loss')
# plt.show()
#
# # Plot train and test accuracy
# plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
# plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
# plt.title('Train and Test Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
#
# # Plot some samples
# # Plot the 6 of the last batch results:
# actuals = rand_y[0:6]
# predictions = np.argmax(temp_train_preds, axis=1)[0:6]
# images = np.squeeze(rand_x[0:6])
#
# Nrows = 2
# Ncols = 3
# for i in range(6):
#     plt.subplot(Nrows, Ncols, i + 1)
#     plt.imshow(np.reshape(images[i], [28, 28]), cmap='Greys_r')
#     plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
#               fontsize=10)
#     frame = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)