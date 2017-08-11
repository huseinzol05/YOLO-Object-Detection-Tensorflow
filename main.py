import settings
import model
from utils import VOC
import os
import time
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
model = model.Model(training = False)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
boundary1 = settings.cell_size * settings.cell_size * settings.num_class
boundary2 = boundary1 + settings.cell_size * settings.cell_size * settings.box_per_cell

try:
    saver.restore(sess, os.getcwd() + '/model.ckpt')
    print 'load from past checkpoint'
except:     
    try:
        saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
        print 'load from YOLO small pretrained'
    except:
        print 'you must train first, exiting..'
        exit(0)

def draw_result(img, result):
    for i in range(len(result)):
        x = int(result[i][1])
        y = int(result[i][2])
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)
        
def detect(img):
    img_h, img_w, _ = img.shape
    inputs = cv2.resize(img, (settings.image_size, settings.image_size))
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
    inputs = (inputs / 255.0) * 2.0 - 1.0
    inputs = np.reshape(inputs, (1, settings.image_size, settings.image_size, 3))
    result = detect_from_cvmat(inputs)[0]
    print result

    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / settings.image_size)
        result[i][2] *= (1.0 * img_h / settings.image_size)
        result[i][3] *= (1.0 * img_w / settings.image_size)
        result[i][4] *= (1.0 * img_h / settings.image_size)

    return result

def detect_from_cvmat(inputs):
    net_output = sess.run(model.logits, feed_dict = {model.images: inputs})
    results = []
    for i in range(net_output.shape[0]):
        results.append(interpret_output(net_output[i]))

    return results

def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

def interpret_output(output):
    probs = np.zeros((settings.cell_size, settings.cell_size, settings.box_per_cell, len(settings.classes_name)))
    class_probs = np.reshape(output[0 : boundary1], (settings.cell_size, settings.cell_size, settings.num_class))
    scales = np.reshape(output[boundary1 : boundary2], (settings.cell_size, settings.cell_size, settings.box_per_cell))
    boxes = np.reshape(output[boundary2 :], (settings.cell_size, settings.cell_size, settings.box_per_cell, 4))
    offset = np.transpose(np.reshape(np.array([np.arange(settings.cell_size)] * settings.cell_size * settings.box_per_cell), [settings.box_per_cell, settings.cell_size, settings.cell_size]), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / settings.cell_size
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

    boxes *= settings.image_size

    for i in range(settings.box_per_cell):
        for j in range(settings.num_class):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= settings.threshold, dtype = 'bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis = 3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > settings.IOU_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype = 'bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append([settings.classes_name[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

    return result

def read_image(image, name):
    result = detect(image)
    draw_result(image, result)
    plt.imshow(image)
    plt.savefig(os.getcwd() + '/' + name + 'output.png')

if settings.output == 1:
    image = cv2.imread(settings.picture_name)
    read_image(image, settings.picture_name[-10:])
    
if settings.output == 2:
    labels = VOC('test').load_labels()
    for i in xrange(len(labels)):
        print labels[i]['imname']
        image = cv2.imread(labels[i]['imname'])
        read_image(image, labels[i]['imname'][-10:])

if settings.output == 3:
    cap = cv2.VideoCapture(-1)
    ret, _ = cap.read()
    while ret:
        ret, frame = cap.read()
        result = detect(frame)
        draw_result(frame, result)
        cv2.imshow('Camera', frame)
        cv2.waitKey(wait)
