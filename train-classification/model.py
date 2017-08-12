import tensorflow as tf
import settings
import numpy as np

slim = tf.contrib.slim

def build_network(self, images, num_outputs, keep_prob = settings.dropout, training = True, scope = 'yolo'):
        with tf.variable_scope(scope):
                with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = tf.nn.relu, 
                                    weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), 
                                    weights_regularizer = slim.l2_regularizer(0.0005)):
                        net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name = 'pad_1')
                        net = slim.conv2d(net, 64, 7, 2, padding = 'VALID', scope = 'conv_2')
                        net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_3')
                        net = slim.conv2d(net, 192, 3, scope = 'conv_4')
                        net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_5')
                        net = slim.conv2d(net, 128, 1, scope = 'conv_6')
                        net = slim.conv2d(net, 256, 3, scope = 'conv_7')
                        net = slim.conv2d(net, 256, 1, scope = 'conv_8')
                        net = slim.conv2d(net, 512, 3, scope = 'conv_9')
                        net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_10')
                        net = slim.conv2d(net, 256, 1, scope = 'conv_11')
                        net = slim.conv2d(net, 512, 3, scope = 'conv_12')
                        net = slim.conv2d(net, 256, 1, scope = 'conv_13')
                        net = slim.conv2d(net, 512, 3, scope = 'conv_14')
                        net = slim.conv2d(net, 256, 1, scope = 'conv_15')
                        net = slim.conv2d(net, 512, 3, scope = 'conv_16')
                        net = slim.conv2d(net, 256, 1, scope = 'conv_17')
                        net = slim.conv2d(net, 512, 3, scope = 'conv_18')
                        net = slim.conv2d(net, 512, 1, scope = 'conv_19')
                        net = slim.conv2d(net, 1024, 3, scope = 'conv_20')
                        net = slim.max_pool2d(net, 2, padding='SAME', scope = 'pool_21')
                        net = slim.conv2d(net, 512, 1, scope = 'conv_22')
                        net = slim.conv2d(net, 1024, 3, scope = 'conv_23')
                        net = slim.conv2d(net, 512, 1, scope = 'conv_24')
                        net = slim.conv2d(net, 1024, 3, scope = 'conv_25')
                        net = slim.conv2d(net, 1024, 3, scope = 'conv_26')
                        net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name = 'pad_27')
                        net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope = 'conv_28')
                        net = slim.conv2d(net, 1024, 3, scope = 'conv_29')
                        net = slim.conv2d(net, 1024, 3, scope = 'conv_30')
                        net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                        net = slim.flatten(net, scope = 'flat_32')
                        net = slim.fully_connected(net, 512, scope = 'fc_33')
                        net = slim.fully_connected(net, 4096, scope = 'fc_34')
                        net = slim.dropout(net, keep_prob = keep_prob, is_training = training, scope = 'dropout_35')
                        net = slim.fully_connected(net, num_outputs, activation_fn = None, scope = 'fc_36')
        return net
class Model:
    
    def __init__(self, label_size):
        self.batch = tf.Variable(0)
        self.images = tf.placeholder(tf.float32, [None, settings.image_size, settings.image_size, 3])
        self.labels = placeholder(tf.float32, [None, label_size])
        self.learning_rate = tf.train.exponential_decay(settings.learning_rate, self.batch * settings.batch_size, settings.decay_step, settings.decay_rate, True)
        self.training = True
        
        self.logits = build_network(self.images, num_outputs = label_size, training = self.training)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        