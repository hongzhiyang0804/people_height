import numpy as np
import tensorflow as tf
from PIL import Image
import time
import utils
import os
import psutil
info = psutil.virtual_memory()
# input_image = './6.jpg'
class_names = './person.names'
input_size = 416
frozen_model = './yolov3_voc_person.pb'
conf_threshold = 0.5
iou_threshold = 0.4
classes = utils.load_coco_names(class_names)

t0 = time.time()
frozenGraph = utils.load_graph(frozen_model)
print("Loaded graph in {:.2f}s".format(time.time()-t0))
sess = tf.Session(graph=frozenGraph)

path = './data/'
new_path = './region/'
j = 0
for i in os.listdir(path):
    file_name = i
    image_name = path + file_name
    new_name = new_path + file_name
    image = Image.open(image_name)
    print(new_name)
    img_resized = utils.letter_box_image(image, input_size, input_size, 128)
    img_resized = img_resized.astype(np.float32)
    boxes, inputs = utils.get_boxes_and_inputs_pb(frozenGraph)

    t0 = time.time()
    detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})
    filtered_boxes = utils.non_max_suppression(detected_boxes,
                                               confidence_threshold=conf_threshold,
                                               iou_threshold=iou_threshold)
    print("Predictions found in {:.2f}s".format(time.time() - t0))
    if filtered_boxes:
        img, region, score, box = utils.draw_boxes(filtered_boxes, image, classes, (input_size, input_size), True)
        # img.show()
        # print()
        if score >= 0.75:
            print(box)
            img.show()
#             # img.save(new_name)
#             region.save(new_name)
#         # region.show()
#     else:
#         j = j + 1
# print(j)

# print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
# print(u'总内存：', info.total)
# print(u'内存占比：', info.percent)
# print(u'cpu个数：', psutil.cpu_count())
