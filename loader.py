import os
import numpy as np
import tensorflow as tf
import time

from processor import Clip

class DataLoader:

    def __init__(self, train_file, tfrecords_filename, feature_vector_size, num_epochs, batch_size, batch_process_threads_num):
        self.batch_process_threads_num = batch_process_threads_num
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.feature_vector_size = feature_vector_size
        self.train_dir = "./"
        self.train_file = train_file
        self.tfrecords_filename = tfrecords_filename
        self.number_of_class = 10

    def load_dataset(self, name):
        """Load all dataset recordings into a nested list."""
        clips = []
        for directory in sorted(os.listdir('{0}/'.format(name))):
            directory = '{0}/{1}'.format(name, directory)
            if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
                print('Parsing ' + directory)
                category = []
                for clip in sorted(os.listdir(directory)):
                    if clip[-3:] == 'ogg':
                        print ('{0}/{1}'.format(directory, clip))
                        category.append(Clip('{0}/{1}'.format(directory, clip), 'ogg'))
                clips.append(category)
        print('All {0} recordings loaded.'.format(name))
        return clips

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def create_one_big_file(self, file_type):
        writer = tf.python_io.TFRecordWriter(self.tfrecords_filename)
        # filename = os.path.join(self.train_dir, self.train_file if self.is_train else self.validation_file)
        for directory in sorted(os.listdir('{0}/'.format(self.train_file))):
            directory = '{0}/{1}'.format(self.train_file, directory)
            if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
                print('Parsing ' + directory)
                for clip in sorted(os.listdir(directory)):
                    if clip[-3:] == 'ogg':
                        print ('{0}/{1}'.format(directory, clip))
                        clip_category =('{0}/{1}'.format(directory, clip), directory.split("/")[1].split("-")[0].strip())[1]
                        raw_data_clip = Clip('{0}/{1}'.format(directory, clip), file_type).get_feature_vector()[0]
                        # TODO add multiple vectors into feature list
                        width = len(raw_data_clip)
                        height = 1
                        raw_data_clip = raw_data_clip.tostring()
                        clip_label = self.get_label(int(clip_category)).tostring()
                        clip_raw = tf.train.Example(features=tf.train.Features(feature={
                            'clip_height': self._int64_feature(height),
                            'clip_width': self._int64_feature(width),
                            'clip_raw': self._bytes_feature(raw_data_clip),
                            'clip_label_raw': self._bytes_feature(clip_label)}))
                        writer.write(clip_raw.SerializeToString())
        writer.close()
        print('All {0} recordings loaded.'.format(self.train_file))

    def get_label(self, class_number):
        label = np.zeros(self.number_of_class, dtype=np.int)
        label[class_number-1] = 1
        return label

    def read_tf_recode(self):
        reconstructed_clips = []
        record_iterator = tf.python_io.tf_record_iterator(path=self.tfrecords_filename)
        for string_record in record_iterator:
            raw_clip = tf.train.Example()
            raw_clip.ParseFromString(string_record)
            height = int(raw_clip.features.feature['clip_height'].int64_list.value[0])
            width = int(raw_clip.features.feature['clip_width'].int64_list.value[0])
            img_string = (raw_clip.features.feature['clip_raw'].bytes_list.value[0])
            label = (raw_clip.features.feature['clip_label_raw'].bytes_list.value[0])
            img_1d = np.fromstring(img_string, dtype=np.float64)
            label = np.fromstring(img_string, dtype=np.uint64)
            reconstructed_clip = img_1d.reshape((height, width, -1))
            reconstructed_clip_label = label.reshape((1, self.number_of_class, -1))
            reconstructed_clips.append((reconstructed_clip, reconstructed_clip_label))
        return reconstructed_clips

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                 'clip_height': tf.FixedLenFeature([], tf.int64),
                 'clip_width': tf.FixedLenFeature([], tf.int64),
                 'clip_raw': tf.FixedLenFeature([], tf.string),
                 'clip_label_raw': tf.FixedLenFeature([], tf.string)
            })
        image = tf.decode_raw(features['clip_raw'], tf.float64)
        label = tf.decode_raw(features['clip_label_raw'], tf.int64)
        image = tf.reshape(image, [1, self.feature_vector_size, 1])
        label = tf.reshape(label, [1, self.number_of_class, 1])
        return image, label


    def inputs(self):
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([self.tfrecords_filename], num_epochs=self.num_epochs)
            image, label = self.read_and_decode(filename_queue)
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=self.batch_size, num_threads=self.batch_process_threads_num,
                capacity=1000 + 3 * self.batch_size,
                min_after_dequeue=100)
            return images, sparse_labels

    def run_training(self):
        with tf.Graph().as_default():
            image, label = self.inputs()
            with tf.Session()  as sess:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                try:
                    step = 0
                    while not coord.should_stop():
                        start_time = time.time()
                        while not coord.should_stop():
                            # Run training steps or whatever
                            example, l = sess.run([image, label])
                            print example
                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (self.num_epochs, self.batch_size))
                finally:
                    coord.request_stop()
                coord.join(threads)
                sess.close()

# loader = DataLoader('TRAIN-10', "audio_clips_segmentation.tfrecords", 512, 1,1, 2)
# loader.create_one_big_file("ogg")
# loader.run_training()


# print "data"




