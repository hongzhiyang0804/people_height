import cv2
import numpy as np
# 去畸变
npzfile = np.load('./calibrate.npz')
mtx = npzfile['mtx']
dist = npzfile['dist']
img = cv2.imread('./image.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
print(roi)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
if roi != (0, 0, 0, 0):
    dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)