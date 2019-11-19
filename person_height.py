import numpy as np
import tensorflow as tf
from PIL import Image
import time
import utils
import cv2
import os
# 本项目实现用户整体检测，获取在图像中的高度像素值，用于根据三角形近似定力计算用户身高,参照物选择尽量选取与用户身高等同或较大物品，所得结果会更为准确。
# input_image = './1/101115_4d9192a403db450da0b400e6116e686a.png'
input_image = './calibresult.png'
class_names = './person.names'
input_size = 416
frozen_model = './yolov3_voc_person.pb'
conf_threshold = 0.5
iou_threshold = 0.4
classes = utils.load_coco_names(class_names)
out_image = './person.jpg'

t0 = time.time()
frozenGraph = utils.load_graph(frozen_model)
print("Loaded graph in {:.2f}s".format(time.time()-t0))
sess = tf.Session(graph=frozenGraph)


# image = cv2.imread(input_image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = Image.fromarray(image.astype('uint8')).convert('RGB')
# 上面三步等同于下面的Image.open()操作
image = Image.open(input_image)
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
    # if len(filtered_boxes[0][:]) == 1:
    img, region, score, box = utils.draw_boxes(filtered_boxes, image, classes, (input_size, input_size), True)
    # box = np.array(box)
    # print(box)
    if score > 0.90:
        person_image_height = box[0][3] - box[0][1]
        img.show()
        region.save(out_image)
        print(person_image_height)
        # 计算当前用户身高
        # 可根据参照物(本例采用椅子作为参照物，其实际高度为94.5cm，在固定距离下该参照物在图像中像素值为392)实际高度与图像高度像素，
        # 获取人物图像像素高度。具体调参需在具体环境下进行调参
        # 此方法存在较大的误差，故结果仅供趣味输出，追求准确仍需具体输入准确值
        person_height = (person_image_height * 96) / 230
        print("person_height: %.2fcm \n" % (person_height))




