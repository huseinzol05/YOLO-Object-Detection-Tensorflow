import tensorflow as tf
import settings

class Model:
    
    def __init(self, training = True):
        self.classes = settings.classes_name
        self.num_classes = len(settings.classes_name)
        self.image_size = settings.image_size
        self.cell_size = settings.cell_size
        self.boxes_per_cell = settings.box_per_cell
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = settings.object_scale
        self.no_object_scale = settings.no_object_scale
        self.class_scale = settings.class_scale
        self.coord_scale = settings.coordinate_scale
        
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(tf.float32, [None, settings.image_size, settings.image_size, 3])
        
        self.logits = self.build_network(self.images, num_outputs = self.output_size, alpha = self.alpha, training = training)
        
        if training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.contrib.losses.get_total_loss()
            self.optimizer = tf.train.AdamOptimizer(settings.learning_rate).minimize(self.total_loss)
        
    def build_network(self, images, num_outputs, alpha, keep_prob = settings.dropout, training = True, scope = 'yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = leaky_relu(alpha), weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), weights_regularizer = slim.l2_regularizer(0.0005)):
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
    
    
    def calc_iou(self, boxes1, boxes2, scope = 'iou'):
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope = 'loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtyp e= tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name = 'class_loss') * self.class_scale

            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name = 'object_loss') * self.object_scale

            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name = 'noobject_loss') * self.no_object_scale

            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name = 'coord_loss') * self.coord_scale

            tf.contrib.losses.add_loss(class_loss)
            tf.contrib.losses.add_loss(object_loss)
            tf.contrib.losses.add_loss(noobject_loss)
            tf.contrib.losses.add_loss(coord_loss)

def leaky_relu(alpha):
    return tf.maximum(alpha * inputs, inputs)