import numpy as np
import tensorflow as tf
import config as cfg

slim = tf.contrib.slim

class ALEXNET(object):

    def __init__(self, is_training=True):
        self.num_class = int(cfg.NC) #CLASSES
        self.image_size = cfg.IMAGE_SIZE
        self.output_size = int(cfg.NC)

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE

        self.momentum = cfg.MOMENTUM

        self.weight_decay = cfg.WEIGHT_DECAY
        self.initial_weights_mean = cfg.INITIAL_WEIGHTS_MEAN
        self.initial_weights_std = cfg.INITIAL_WEIGHTS_STD
        self.initial_bias = cfg.INITIAL_BIAS
        self.keep_prob = tf.placeholder(tf.float32)
     
        # Local Response Normalization parameters
        self.radius = 5; self.alpha = 0.0001; self.beta = 0.75; self.bias = 1.0

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, is_training=is_training) # Output after feed-forward on the network
        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.num_class])

    # NETWORK CONFIGURATION
    def build_network(self,
                      images,
                      num_outputs=483,
                      keep_prob=0.5,
                      is_training=True,
                      scope='alexnet' ): 
        with tf.variable_scope(scope):
            with slim.arg_scope( [slim.conv2d, slim.fully_connected],
                                #activation_fn=tf.nn.relu,
                                biases_initializer=tf.constant_initializer(self.initial_bias), 
                                biases_regularizer=slim.l2_regularizer(0.0), 
                                weights_initializer=tf.truncated_normal_initializer(self.initial_weights_mean, self.initial_weights_std),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay) ):

                print(images.shape)
                # conv1 - pool1 - LRN
                net = slim.conv2d(images, 96, 11, 4, padding='VALID', biases_initializer=tf.constant_initializer(0.0), activation_fn=tf.nn.relu, scope='conv_1') 
                # 96 filtres - taille 11x11 - stride [4 4] - padding VALID
                print(net.shape)
                net = slim.max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool_1') 
                # taille 3x3 - stride [2 2] - padding VALID (default)
                net = tf.nn.local_response_normalization(net, depth_radius=self.radius, alpha=self.alpha, beta=self.beta, bias=self.bias, name='norm1') 
                print(net.shape)

                # conv2 - pool2 - LRN
                net = slim.conv2d(net, 256, 5, 1, activation_fn=tf.nn.relu, scope='conv_2') 
                # 256 filtres - taille 5x5 - stride [1 1] - padding SAME (default)
                print(net.shape)
                net = slim.max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool_2')
                # taille 3x3 - stride [2 2] - padding VALID (default)
                net = tf.nn.local_response_normalization(net, depth_radius=self.radius, alpha=self.alpha, beta=self.beta, bias=self.bias, name='norm2')
                print(net.shape)

                # conv3
                net = slim.conv2d(net, 384, 3, 1, biases_initializer=tf.constant_initializer(0.0), activation_fn=tf.nn.relu, scope='conv_3')
                # 384 filtres - taille 3x3 - stride [1 1] - padding SAME (default)
                print(net.shape)

                # conv4
                net = slim.conv2d(net, 384, 3, 1, activation_fn=tf.nn.relu, scope='conv_4')
                # 384 filtres - taille 3x3 - stride [1 1] - padding SAME (default)
                print(net.shape)

                # conv5 - pool5
                net = slim.conv2d(net, 256, 3, 1, activation_fn=tf.nn.relu, scope='conv_5')
                # 256 filtres - taille 3x3 - stride [1 1] - padding SAME (default)
                print(net.shape)
                net = slim.max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool_5')
                # taille 3x3 - stride [2 2] - padding VALID (default)
                print(net.shape)
                net = slim.flatten(net, scope='flat_5')
                print(net.shape)

                # FC-6 - Dropout
                net = slim.fully_connected(net, 2048, activation_fn=tf.nn.relu, scope='fc_6')
                net = tf.nn.dropout(net, self.keep_prob, name='dropout_6')
                print(net.shape)

                # FC-7 - Dropout
                net = slim.fully_connected(net, 2048, activation_fn=tf.nn.relu, scope='fc_7')
                #net = slim.fully_connected(net, 2048, activation_fn=tf.nn.relu, scope='fc_7')
                net = tf.nn.dropout(net, self.keep_prob, name='dropout_7')
                print(net.shape)

                # FC-8
                net = slim.fully_connected(net, self.output_size, activation_fn=None, scope='fc_8') 
                print(net.shape)
        return net
