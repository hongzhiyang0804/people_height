import cv2
import numpy as np
import glob
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
Nx_cor = 9
Ny_cor = 6

objp = np.zeros((Nx_cor * Ny_cor, 3), np.float32)
objp[:, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

count = 0  # count 用来标志成功检测到的棋盘格画面数量
images = glob.glob('./image/*.jpg')
for fname in images:
    print(fname)
    img = cv2.imread(fname)

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)  # Find the corners
    # If found, add object points, image points
    if ret == True:
        print(ret)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (Nx_cor, Ny_cor), corners, ret)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx, dist)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: ", mean_error / len(objpoints))
np.savez('calibrate.npz', mtx=mtx, dist=dist[0:4])