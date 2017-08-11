import settings
import model
from utils import VOC
import os
import time
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
model = model.Model()
utils = VOC('train')
saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'yolo'))
sess.run(tf.global_variables_initializer())
    
try:
    saver.restore(sess, os.getcwd() + '/model.ckpt')
    print 'load from past checkpoint'
except:     
    try:
        saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
        print 'load from YOLO small pretrained'
    except:
        print 'exit, atleast need a pretrained model'
        exit(0)
            
for i in xrange(settings.epoch):
        
    last_time = time.time()
    total_loss = 0
    
    for x in xrange(0, len(utils.gt_labels) - settings.batch_size, settings.batch_size):
        images = np.zeros((settings.batch_size, settings.image_size, settings.image_size, 3))
        labels = np.zeros((settings.batch_size, settings.cell_size, settings.cell_size, 25))
        
        for n in xrange(settings.batch_size):
            imname = utils.gt_labels[x + n]['imname']
            flipped = utils.gt_labels[x + n]['flipped']
            images[n, :, :, :] = utils.image_read(imname, flipped)
            labels[n, :, :, :] = utils.gt_labels[x + n]['label']
        
        learning_rate = tf.train.exponential_decay(settings.learning_rate, ((i + 1) * x), settings.decay_step, settings.decay_rate, True)
        loss, _ = sess.run([model.total_loss, model.optimizer], feed_dict = {model.images: images, model.labels: labels, models.learning_rate: learning_rate})
        total_loss += loss

        if (x + 1) % settings.checkpoint == 0:
            print 'checkpoint reached: ' + str(x + 1)
    
    np.random.shuffle(utils.gt_labels)
    print 'epoch: ' + str(i + 1) + ', loss: ' + str(loss / (len(utils.gt_labels) - settings.batch_size / (settings.batch_size * 1.0))) + ',  s / epoch: ' + str(time.time() - last_time)
    saver.save(sess, os.getcwd() + '/model.ckpt')
        
    
