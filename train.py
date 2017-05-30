import settings
import model
from utils import VOC
import os
import time
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = settings.memory_duringtesting
sess = tf.InteractiveSession(config = config)
model = model.Model()
utils = VOC('train')
saver = tf.train.Saver(tf.global_variables(), max_to_keep = None)
sess.run(tf.global_variables_initializer())
    
try:
    saver.restore(sess, os.getcwd() + '/model.ckpt')
    print 'load from past checkpoint'
except:     
    try:
        saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
        print 'load from YOLO small pretrained'
    except:
        print 'start from fresh variables'
            
for i in xrange(settings.epoch):
        
    last_time = time.time()
    total_loss = 0
    
    print len(utils.gt_labels)
    
    for x in xrange(0, len(utils.gt_labels) - settings.batch_size, settings.batch_size):
        images = np.zeros((settings.batch_size, settings.image_size, settings.image_size, 3))
        labels = np.zeros((settings.batch_size, settings.cell_size, settings.cell_size, 25))
        
        for n in xrange(settings.batch_size):
            imname = utils.gt_labels[x + n]['imname']
            flipped = utils.gt_labels[x + n]['flipped']
            images[n, :, :, :] = utils.image_read(imname, flipped)
            labels[n, :, :, :] = utils.gt_labels[x + n]['label']
            
        loss, _ = sess.run([model.total_loss, model.optimizer], feed_dict = {model.images: images, model.labels: labels})
        total_loss += loss

        if (x + 1) % settings.checkpoint == 0:
            print 'checkpoint reached: ' + str(x + 1)
    
    np.random.shuffle(utils.gt_labels)
    print 'epoch: ' + str(i + 1) + ', loss: ' + str(loss / (len(utils.gt_labels) - settings.batch_size / (settings.batch_size * 1.0))) + ',  s / epoch: ' + str(time.time() - last_time)
    saver.save(sess, os.getcwd() + '/model.ckpt')
        
    