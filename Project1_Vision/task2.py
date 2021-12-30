###############
# Design the function "calibrate" to  return
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates.
#                                            True if the intrinsic parameters are invariable.
# It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    # ......
    chessboardSize = (4, 9)

    img = imread(imgname)

    img_gray = cvtColor(img, COLOR_BGR2GRAY)
    height, width, channels = img.shape
    #print(height, width, channels)
   
    found, corners = findChessboardCorners(
        img_gray, chessboardSize, None)

    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cornerSubPix(
        img_gray, corners, (11,11), (height, width), criteria)
    img = drawChessboardCorners(img, chessboardSize, corners, found)
    # 2D image coordinates
    imgCoordinates = []  
    imgCoordinates.append(np.vstack(corners))

    #cv2.imshow('Frame', img)
    XYZ = [[40, 0, 0], [40, 0, 10], [40, 0, 20], [40, 0, 30],
           [30, 0, 0], [30, 0, 10], [30, 0, 20], [30, 0, 30],
           [20, 0, 0], [20, 0, 10], [20, 0, 20], [20, 0, 30],
           [10, 0, 0], [10, 0, 10], [10, 0, 20], [10, 0, 30],
           [0, 0, 0], [0, 0, 10], [0, 0, 20], [0, 0, 30],
           [0, 10, 0], [0, 10, 10], [0, 10, 20], [0, 10, 30],
           [0, 20, 0], [0, 20, 10], [0, 20, 20], [0, 20, 30],
           [0, 30, 0], [0, 30, 10], [0, 30, 20], [0, 30, 30],
           [0, 40, 0], [0, 40, 10], [0, 40, 20], [0, 40, 30]
           ]

    n = len(imgCoordinates[0])
    A = np.zeros((2*n, 12))
    #print(imgCoordinates)
    for i in range(n):
        # image coordinates
        x, y = imgCoordinates[0][i]  
        # world coordinates
        X, Y, Z = XYZ[i]  
        # 1 pair of rows in M for one pair of img and world coordinates
        row1 = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        row2 = np.array([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
        A[2*i] = row1
        A[(2*i) + 1] = row2
    
    #print(A)
    u, s, vt = np.linalg.svd(A)
    #print("Computing SVD of M")
    # print(u)
    # print(s)
    # print(vt)
    # last row of vt is x
    x = vt[-1]
    x3Row = x[-4:-1]

    # lambda is srt of sum of srqs of first 3 elements of last row of x
    lambdaI = 1/np.sqrt(np.sum(x3Row**2))

    M = lambdaI*x
    M = M.reshape(3, 4)
    #print("---------------VERIFY Am=0-----------------")
    #print("Verify")
   # print(np.dot(A,x))
    ox = np.sum(np.dot(M[0, 0:3].transpose(), M[2, 0:3]))
    #print("ox is ",ox)

    oy = np.sum(np.dot(M[1, 0:3].transpose(), M[2, 0:3]))
    #print("oy is ",oy)

    fx = np.sqrt(np.dot(M[0, 0:3].transpose(), M[0, 0:3])-ox**2)
   
    fy = np.sqrt(np.dot(M[1, 0:3].transpose(), M[1, 0:3])-oy**2)
   
    intrinsicMatrix = [[fx, 0, ox], [0, fy, oy], [0, 0, 1]]
   
    # return array of intrinstic values and is constant as true
    return [fx, fy, ox, oy], True


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)
