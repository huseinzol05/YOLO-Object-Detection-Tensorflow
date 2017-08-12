import settings
import model
import os
import time
import numpy as np
import tensorflow as tf
import from scipy import misc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

def get_dataset():
    
    list_folder = os.listdir('data/')
    list_images = []
    for i in xrange(len(list_folder)):
        images = os.listdir('data/' + list_folder[i])
        for x in xrange(len(images)):
            image = [list_folder[i] + '/' + images[x], list_folder[i]]
            list_images.append(image)
    list_images = np.array(list_images)
    np.random.shuffle(list_images)
    
    print "before cleaning got: " + str(list_images.shape[0]) + " data"
    
    list_temp = []
    for i in xrange(list_images.shape[0]):
        image = misc.imread('data/' + list_images[i, 0])
        if len(image.shape) < 3:
            continue
        list_temp.append(list_images[i, :].tolist())
        
    list_images = np.array(list_temp)
    print "after cleaning got: " + str(list_images.shape[0]) + " data"
    label = np.unique(list_images[:, 1]).tolist()
    list_images[:, 1] = LabelEncoder().fit_transform(list_images[:, 1])
    return list_images, np.unique(list_images[:, 1]).shape[0], label

data, output_dimension, label = utils.get_dataset()
data, data_test = train_test_split(data, test_size = 0.2)

sess = tf.InteractiveSession()
model = model.Model(output_dimension)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

try:
    saver.restore(sess, os.getcwd() + "/model.ckpt")
    print "load model.."
except:
    print 'start from fresh variables'
    
for i in xrange(settings.epoch):
    total_cost, total_accuracy, last_time, model.train = 0, 0, time.time(), True
    for k in xrange(0, (data.shape[0] // settings.batch_size) * settings.batch_size, settings.batch_size):
        emb_data = np.zeros((settings.batch_size, settings.image_size, settings.image_size, 3), dtype = np.float32)
        emb_data_label = np.zeros((settings.batch_size, output_dimension), dtype = np.float32)
        
        for x in xrange(settings.batch_size):
            image = misc.imread(location + data[k + x, 0])
            image = misc.imresize(image, (settings.image_size, settings.image_size))
            emb_data_label[x, int(data[k + x, 1])] = 1.0
            emb_data[x, :, :, :] = image
            
        _, loss = sess.run([model.optimizer, model.cost], feed_dict = {model.X : emb_data, model.Y : emb_data_label})
        accuracy = sess.run(model.accuracy, feed_dict = {model.X : emb_data, model.Y : emb_data_label})
        total_cost += loss
        total_accuracy += accuracy
        
    total_cost /= (data.shape[0] // settings.batch_size)
    total_accuracy /= (data.shape[0] // settings.batch_size)
    print "epoch: " + str(i + 1) + ", loss: " + str(loss) + ", accuracy: " + str(accuracy) + ", s / epoch: " + str(time.time() - last_time)
    
    model.train, total_accuracy, total_logits = False, 0, []
    for k in xrange(0, (data_test.shape[0] // settings.batch_size) * settings.batch_size, settings.batch_size):
        emb_data = np.zeros((settings.batch_size, settings.image_size, settings.image_size, 3), dtype = np.float32)
        emb_data_label = np.zeros((settings.batch_size, output_dimension), dtype = np.float32)
        
        for x in xrange(settings.batch_size):
            image = misc.imread(location + data_test[k + x, 0])
            image = misc.imresize(image, (settings.image_size, settings.image_size))
            emb_data_label[x, int(data_test[k + x, 1])] = 1.0
            emb_data[x, :, :, :] = image
            
        accuracy, logits = sess.run([model.accuracy, tf.cast(tf.argmax(model.logits, 1), tf.int32)], feed_dict = {model.X : emb_data, model.Y : emb_data_label})
        total_accuracy += accuracy
        total_logits += logits.tolist()
    
    total_accuracy /= (data_test.shape[0] // settings.batch_size)
    print 'testing accuracy: ' + str(total_accuracy)
    print(metrics.classification_report(data_test, np.array(total_logits), target_names = label))
    
    saver.save(sess, os.getcwd() + '/model.ckpt')