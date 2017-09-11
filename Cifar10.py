#!/usr/bin/env python3
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from Train import Train
from resnet import inference_small
from SeluResNet import *
import tensorflow as tf

class Cifar10:

    numClasses = 10
    dataDir = '/media/HD/datasets'
    use_bn = True

    def __init__(self, selu, batchSize=16, trainExamplePerEpoch=50000, evalExamplePerEpoch=10000, imageSize=32):
        self.selu = selu
        self.batchSize = batchSize
        self.trainExamplePerEpoch = trainExamplePerEpoch
        self.evalExamplePerEpoch = evalExamplePerEpoch 
        self.imageSize = imageSize

    def read_cifar10(self, filename_queue):
        """Reads and parses examples from CIFAR10 data files.

      Recommendation: if you want N-way read parallelism, call this function
      N times.  This will give you N independent Readers reading different
      files & positions within those files, which will give better mixing of
      examples.

      Args:
        filename_queue: A queue of strings with the filenames to read from.

      Returns:
        An object representing a single example, with the following fields:
          height: number of rows in the result (32)
          width: number of columns in the result (32)
          depth: number of color channels in the result (3)
          key: a scalar string Tensor describing the filename & record number
            for this example.
          label: int32 Tensor with shape=(batchSize, numClasses) representing labels
          uint8image: a [height, width, depth] uint8 Tensor with the image data
      """

        class CIFAR10Record(object):
            pass

        result = CIFAR10Record()

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        label_bytes = 1  # 2 for CIFAR-100
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.slice(record_bytes, [label_bytes], [image_bytes]),
            [result.depth, result.height, result.width])
        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def _generate_image_and_label_batch(self, image, label, min_queue_examples, shuffle):
        """Construct a queued batch of images and labels.

      Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batchSize: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

      Returns:
        images: Images. 4D tensor of [batchSize, height, width, 3] size.
        labels: Labels. 1D tensor of [batchSize] size.
      """
        # Create a queue that shuffles the examples, and then
        # read 'batchSize' images + labels from the example queue.
        num_preprocess_threads = 16
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=self.batchSize,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * self.batchSize,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=self.batchSize,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * self.batchSize)

        return images, tf.one_hot(tf.reshape(label_batch, [self.batchSize]), self.numClasses, 1, 0, axis = -1)

    def getInputs(self, eval_data, distort=False, shuffle=False):
        """Construct input for CIFAR evaluation using the Reader ops.

      Args:
        eval_data: bool, indicating if one should use the train or eval data set.

      Returns:
        images: Images. 4D tensor of [batchSize, imageSize, imageSize, 3] size.
        labels: Labels. 1D tensor of [batchSize] size.
      """
        if not eval_data:
            filenames = [
                    os.path.join(self.dataDir, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i)
                    for i in range(1, 6)
            ]
            num_examples_per_epoch = self.trainExamplePerEpoch
        else:
            filenames = [os.path.join(self.dataDir, 'cifar-10-batches-bin', 'test_batch.bin')]
            num_examples_per_epoch = self.evalExamplePerEpoch

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self.read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        width = height = self.imageSize

        if distort:
            # Image processing for training the network. Note the many random
            # distortions applied to the image.

            # Randomly crop a [height, width] section of the image.
            distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
            resized_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        else:
            # Image processing for evaluation.
            # Crop the central [height, width] of the image.
            resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(float_image,
                                               read_input.label,
                                               min_queue_examples,
                                               shuffle=shuffle)

    def buildSeluModel(self, x, numBlocks=3):
        with tf.variable_scope('scale1'):
            x = convSeluResBlock(x, 16, name='scale1Conv')
            x = selu(x)
            for i in range(numBlocks):
                x = selu(convSeluResBlock(x, 16, name='scale1Conv'+str(i)))

        with tf.variable_scope('scale2'):
            x = selu(convSeluResBlock(x, 32, 2, name='scale2Conv0'))
            for i in range(numBlocks-1):
                x = selu(convSeluResBlock(x, 32, name='scale2Conv'+str(i+1)))

        with tf.variable_scope('scale3'):
            x = selu(convSeluResBlock(x, 64, 2, name='scale3Conv0'))
            for i in range(numBlocks-1):
                x = selu(convSeluResBlock(x, 64, name='scale3Conv'+str(i+1)))

        x = tf.reduce_mean(x, reduction_indices=[1,2], name='ave_pool')
        with tf.variable_scope('fc'):
            x = fcSeluResBlock(x, self.numClasses, name="LastLayer")
        return x


    def getModel(self, images):
        if self.selu:
            return self.buildSeluModel(images)
        else:
            return inference_small(images, num_classes=self.numClasses, use_bias=(not self.use_bn), num_blocks=3)

def main(vargs = None):
    c = Cifar10(True, batchSize=32)
    train = Train()
    images_train, labels_train = c.getInputs(False, True, True)
    images_val, labels_val = c.getInputs(True)
    is_training = tf.placeholder('bool', [], name='is_training')

    images, labels = tf.cond(is_training,
        lambda: (images_train, labels_train),
        lambda: (images_val, labels_val))
    logits = c.getModel(images)
    train.train(is_training, logits, images, labels)

if __name__ == '__main__': #TODO fix this
    tf.app.run()
