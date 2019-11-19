import cv2
import numpy as np
import glob
# 该代码采用张友正相机标定方法，使用棋盘格对相机进行标定取畸变，得到准确图像准确位置坐标。该方法存在使用不同舌像头拍摄对最终去畸变的图像产生较大误差。
# 故选择不同的摄像头对结果影响很大。
# 找棋盘格角点
# 阈值
# 设置终止条件，迭代30次或移动0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
w = 9
h = 6
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h, 3), np.float32)
objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('./image/*.jpg')
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print(ret)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.waitKey(2000)
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 去畸变
img2 = cv2.imread('./2019-11-13-184452.jpg')
h,  w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h)) # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# 根据前面ROI区域裁剪图片
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
'''
反向投影误差，我们可以利用反向投影误差对我们找到的参数的准确性评估，
得到的结果越接近0越好，有了内部参数、畸变参数和旋转变化矩阵，
就可以使用cv2.projectPoints()将对象转换到图像点
然后就可以计算变换得到的图像与角点检测算法的绝对差了
最后计算所有标定图像的误差平均值
'''
# 反投影误差
total_error = 0
# 反投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print("total error: ", total_error/len(objpoints))




