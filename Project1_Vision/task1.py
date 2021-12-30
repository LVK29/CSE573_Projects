###############
# Design the function "findRotMat" to  return
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz
# It is ok to add other functions if you need
###############

import numpy as np
import cv2


def findRotMat(alpha, beta, gamma):
    # As numpy trig functions use radians
    gamma1 = np.radians(gamma)
    beta1 = np.radians(beta)
    alpha1 = np.radians(alpha)
    
    RzA = np.array([[np.cos(alpha1), -np.sin(alpha1),  0],
                    [np.sin(alpha1), np.cos(alpha1),   0],
                    [0,                   0,          1]])
    RxB = np.array([[1, 0, 0],
                    [0, np.cos(beta1), -np.sin(beta1)],
                    [0, np.sin(beta1), np.cos(beta1)]])

    RzG = np.array([[np.cos(gamma1), -np.sin(gamma1),  0],
                    [np.sin(gamma1), np.cos(gamma1),   0],
                    [0,                   0,          1]])
    rotMat1 = np.dot(RzG,np.dot( RxB,RzA))
    nRzA=RzA.transpose()
    nRxB=RxB.transpose()
    nRzG=RzG.transpose()
    rotMat2 = np.dot(nRzA,np.dot(nRxB, nRzG))
    return(rotMat1, rotMat2)


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)

   