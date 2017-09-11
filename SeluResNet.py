#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def buildImageNetModel(x, numClasses=1000, numBlocks=[3, 4, 6, 3]):
    with tf.variable_scope('scale1'):
        x = selu(convSeluResBlock(x, 64, 'scale1Conv'))

        with tf.variable_scope('scale2'):
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            for i in range(numBlocks[0]):
                x = selu(convSeluResBlock(x, 64), name='scale3Conv'+str(i))

        with tf.variable_scope('scale3'):
            x = selu(convSeluResBlock(x, 128, 2, name="scale3Conv0"))
            for i in range(numBlocks[1]-1):
                x = selu(convSeluResBlock(x, 128, name="scale3Conv"+str(i+1)))

        with tf.variable_scope('scale4'):
            x = selu(convSeluResBlock(x, 256, 2, name="scale4Conv0"))
            for i in range(numBlocks[2]-1):
                x = selu(convSeluResBlock(x, 256, name="scale4Conv"+str(i+1)))

        with tf.variable_scope('scale5'):
            x = selu(convSeluResBlock(x, 512, 2, name="scale5Conv0"))
            for i in range(numBlocks[1]-1):
                x = selu(convSeluResBlock(x, 512, name="scale5Conv"+str(i+1)))

        x = tf.reduce_mean(x, reduction_indices=[1,2], name="ave_pool")
        with tf.variable_scope('fc'):
            x = fcSeluResBlock(x, numClasses, name="LastLayer")
        return x

# x.get_shape() = [batchSize, length, channels]
# if outChannels=None, the result of this function will be a tensor with the same shape as x
# if innerUnits=None, the hidden layer between the fc layers will be equal to x.get_shape()[1]
def fcSeluResBlock(x, outChannels=None, innerUnits=None, name=""):
    with tf.name_scope(name):
        shape1 = [x.get_shape().as_list()[-1], x.get_shape().as_list()[-2]]
        if innerUnits != None:
            shape1[-1] = innerUnits
        shape2 = list(reversed(shape1))
        if outChannels != None:
            shape2[-1] = outChannels
        fc = selu(fcLayer(x, shape1, name=name+"_fc1"))
        fc = fcLayer(fc, shape2, name=name+"_fc2")
        if outChannels != None:
            shortcut = fcLayer(x, [x.get_shape().as_list()[-1], outChannels], name=name+"_shortcut")
            return tf.add(shortcut, fc)
        else:
            return tf.add(x, fc)

def fcLayer(x, shape, name=""):
    with tf.variable_scope(name):
        w = tf.get_variable(name+"_weights", shape, tf.float32, tf.random_normal_initializer(stddev=np.sqrt(1/shape[0])))
        b = tf.get_variable(name+"_biases", shape[-1], tf.float32, tf.zeros_initializer())
        return tf.nn.bias_add(tf.matmul(x, w), b)

def selu(x, alpha=1.6732632423543772848170429916717, lamb=1.0507009873554804934193349852946):
    return lamb*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def convSeluResBlock(x, outChannels=None, strides=1, name=""):
    with tf.name_scope(name):
        if outChannels == None:
            outChannels = x.get_shape().as_list()[-1]
        conv = selu(convLayer(x, (3,3), outChannels, strides, name=name+"_conv1"))
        # notice strides is left out here, just like paper
        conv = convLayer(conv, (3,3), outChannels, name=name+"_conv2")
        if outChannels == x.get_shape().as_list()[-1]:
            return tf.add(x, conv)
        else:
            shortcut = convLayer(x, (1,1), outChannels, strides, name=name+"_shortcut")
            return tf.add(shortcut, conv)


def convLayer(x, filtShape, outChannels, strides=1, name=""):
    with tf.variable_scope(name):
        inChannels = x.get_shape().as_list()[-1]
        w = tf.get_variable(name+"_weights", (filtShape[0], filtShape[1], inChannels, outChannels), tf.float32, tf.random_normal_initializer(stddev=np.sqrt(1/inChannels)))
        b = tf.get_variable(name+"_biases", outChannels, tf.float32, tf.zeros_initializer())
        return tf.nn.bias_add(tf.nn.conv2d(x, w, [1, strides, strides, 1], padding="SAME"), b)
