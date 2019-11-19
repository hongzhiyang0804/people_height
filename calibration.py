import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import time
import utils
# yolov3检测用户
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
# yolov3检测
def detection(path):
    image = Image.open(path)
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
            # region.save(out_image)
            print(person_image_height)
            # 计算当前用户身高
            # 可根据参照物(本例采用椅子作为参照物，其实际高度为96cm，在固定距离下该参照物在图像中像素值为230)实际高度与图像高度像素，
            # 获取人物图像像素高度。具体调参需在具体环境下进行调参
            # 此方法存在较大的误差，故结果仅供趣味输出，追求准确仍需具体输入准确值
            person_height = (person_image_height * 96) / 230
            print("person_height: %.2fcm \n" % (person_height))

# 相机棋盘图标定
def calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    Nx_cor = 9
    Ny_cor = 6
    objp = np.zeros((Nx_cor * Ny_cor, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    count = 0  # count 用来标志成功检测到的棋盘格画面数量
    while (True):
        ret, frame = cap.read()
        # print(frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)  # Find the corners
            # If found, add object points, image points
            if ret == True:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ret)
                count += 1
                if count > 20:
                    break
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    global mtx, dist
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print(mtx, dist)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: ", mean_error / len(objpoints))
        # # When everything done, release the capture
    np.savez('calibrate.npz', mtx=mtx, dist=dist[0:4])
# 去畸变
def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # print(roi)
    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]
    return dst

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    mtx = []
    dist = []
    # 若本地存在相机定位与校正的相关参数模型(calibrate.npz)，直接调用该模型：
    try:
        npzfile = np.load('calibrate.npz')
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    # 若不存在则进行相机棋盘定位与校正保存对应参数模型
    except IOError:
        calibrate()
    # print('dist', dist[0:4])
    while(True):
        # 读取摄像头每一帧图像
        ret, frame = cap.read()
        #　对每帧图像进行去畸变处理
        frame = undistortion(frame, mtx, dist[0:4])
        # Display the resulting frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        if k == ord('s'):  # 若检测到按键 ‘s’，保存该帧图片，并关闭摄像头
            cv2.imwrite('calibresult.png', frame)
            break
    # yolov3检测
    detection('calibresult.png')

    cap.release()
    cv2.destroyAllWindows()
