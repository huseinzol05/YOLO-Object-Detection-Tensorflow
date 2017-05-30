classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes_no = [i for i in xrange(len(classes_name))]
classes_dict = dict(zip(classes_name, classes_no))
num_class = len(classes_name)

image_size = 448
cell_size = 7
box_per_cell = 2
alpha_relu = 0.1
object_scale = 2.0
no_object_scale = 1.0
class_scale = 2.0
coordinate_scale = 5.0
flipped = True

memory_duringtraining = 0.8
memory_duringtesting = 0.8
learning_rate = 0.001
dropout = 0.5
batch_size = 3
epoch = 15000
checkpoint = 10

# For main
threshold = 0.2
IOU_threshold = 0.5
test_percentage = 0.05

# 1 for read a picture
# 2 to read from testing dataset
# 3 to read from webcam / video
output = 2
# let empty if want to capture from webcam
picture_name = ''
video_name = ''


