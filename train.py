import settings
import model
from utils import VOC
import os
import time
import tensorflow as tf

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
    images, labels = utils.get()
        
    loss, _ = sess.run([model.total_loss, model.optimizer], feed_dict = {model.images: images, model.labels: labels})
        
    if (i + 1) % settings.checkpoint == 0:
        print 'epoch: ' + str(i + 1) + ', loss: ' + str(loss) + ',  s / epoch: ' + str(time.time() - last_time)
        saver.save(sess, os.getcwd() + '/model.ckpt')
        
    