#
# Import
#
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
from preprocessing import inception_preprocessing


#
# Alias
#
slim = tf.contrib.slim
TFDecoder = slim.tfexample_decoder.TFExampleDecoder
preprocess_image = inception_preprocessing.preprocess_image
DataProvider = slim.dataset_data_provider.DatasetDataProvider


#
# Class
#
class ImageDataset:
    
    def __init__(self, base_path, record_name, purpose='train', 
                 shuffle=True, num_classes=None, width=299, height=299, 
                 num_thread=16, num_readers=1, queue_min=24, batch_size=16):

        self.base_path = base_path
        self.record_name = record_name
        self.purpose = purpose

        self.num_classes = num_classes # |labels|
        self.image_width, self.image_height = width, height

        self.shuffle = shuffle
        self.num_thread = num_thread
        self.num_readers = num_readers

        # Dependent calculated
        self.queue_min = queue_min
        self.batch_size = batch_size

        # Dataset 
        self.dataset = None
        self.labels_to_name = None


    #
    # Core API(s)
    #
    def load(self):
        self.labels_to_name = self._load_label()

        num_classes = self.num_classes or len(self.labels_to_name)
        records_path = self.get_record_paths()
        num_samples = self.get_num_samples(records_path)

        self.dataset = slim.dataset.Dataset(
                reader=tf.TFRecordReader,
                decoder=self.record_decoder(),
                data_sources=records_path,
                num_readers=self.num_thread,
                num_samples=num_samples,
                num_classes=num_classes,
                labels_to_name=self.labels_to_name,
                items_to_descriptions=self.features_description())

        self.data_provider = self.make_data_provider()
        return # return success / fail code

    def _load_label(self):
        labels = open(self.label_path(), 'r')

        labels_to_class = {}
        for line in labels:
            label, name = line.split(':')
            labels_to_class[int(label)] = name[:-1]

        return labels_to_class

    def get_batch(self, dynamic_pad=False):
        raw_image, label = self.data_provider.get(['image', 'label'])

        height, width = self.image_height, self.image_width
        origin_image = self.resize(raw_image, height, width)
        image = preprocess_image(raw_image, height, width, self.is_train())

        tensors = [image, origin_image, label]
        images, origin_images, labels = tf.train.batch(
            tensors,
            self.batch_size, 
            self.num_thread,
            self.queue_capacity(),
            allow_smaller_final_batch=True)

        return images, origin_images, labels

    #
    # Image process
    #
    def resize(self, images, height, width):
        images = tf.expand_dims(images, 0)
        images = tf.image.resize_nearest_neighbor(images, [height, width])
        images = tf.squeeze(images)
        
        return images

    #
    # Dataset & Feature property
    #
    def make_data_provider(self):
        return DataProvider(self.dataset, 
                shuffle=self.shuffle,
                num_readers=self.num_readers,
                common_queue_min=self.queue_min,
                common_queue_capacity=self.queue_capacity())

    def record_decoder(self):
        return TFDecoder(self.features(), self.features_decoder())

    def features(self):
        return {
            'image/encoded': str_fixed_len_feature(''),
            'image/format':  str_fixed_len_feature('jpg'),
            'image/class/label': int64_fixed_len_feature([])  
        }

    def features_decoder(self):
        return {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

    def features_description(self):
        return { 
            'image': '%s images' % self.record_name, 
            'label': 'label(id:name).' 
        }


    #
    # Accessor(s)
    #
    def get_num_classes(self):
        return self.dataset.num_classes

    def num_steps_per_epoch(self):
        return int(self.dataset.num_samples / self.batch_size)

    #
    # Helper
    #
    def is_train(self):
        return 'train' in self.purpose

    def label_path(self):
        return os.path.join(self.base_path, 'labels.txt')

    def record_path(self, filename):
        return os.path.join(self.base_path, filename)

    def is_target(self, filename):
        return filename.startswith(self.record_name + '_' + self.purpose)

    def get_record_paths(self):
        paths = [self.record_path(file) for file 
                in os.listdir(self.base_path) if self.is_target(file)]
        return paths

    def get_num_samples(self, paths):
        num_samples = 0
        for record in paths:
            for sample in tf.python_io.tf_record_iterator(record):
                num_samples += 1
        return num_samples

    def queue_capacity(self):
        return 3*self.batch_size + self.queue_min


#
# Feature Helper
#
def str_fixed_len_feature(str_val):
    return tf.FixedLenFeature((), tf.string, default_value=str_val)

def int64_fixed_len_feature(int64_arr):
    return tf.FixedLenFeature(int64_arr, tf.int64, 
            default_value=tf.zeros(int64_arr, dtype=tf.int64))
