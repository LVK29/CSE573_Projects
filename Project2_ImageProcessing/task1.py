"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def matchKNN(descriptors1, descriptors2):
    """
    KNN matches are calcualted for the image descriptors

    Parameters
    ----------
    descriptors1 : TYPE
        DESCRIPTION.
    descriptors2 : TYPE
        DESCRIPTION.

    Returns
    -------
    knnmatches : TYPE
        DESCRIPTION.

    """
    
    knnmatches = []
    for i in range(descriptors1.shape[0]):
        distance = []
        for j in range(descriptors2.shape[0]):
            distance.append(
                np.linalg.norm(descriptors1[i]-descriptors2[j]))
        distance_vector = np.asarray(distance)
        distance_sorted = np.sort(distance_vector)
       
        dMatch1 = cv2.DMatch(i, np.argwhere(distance_vector ==  distance_sorted[0]), distance_sorted[0])
        dMatch2 = cv2.DMatch(i, np.argwhere(distance_vector ==  distance_sorted[1]), distance_sorted[1])
        knnmatches.append([dMatch1, dMatch2])
    return knnmatches



def crossCheck(knnmatchesLtoR, knnmatchesRtoL):
    """
    crossCheck is performed for both the combinations of matches

    Parameters
    ----------
    knnmatchesLtoR : TYPE
        DESCRIPTION.
    knnmatchesRtoL : TYPE
        DESCRIPTION.

    Returns
    -------
    filteredMatches : TYPE
        DESCRIPTION.

    """
    
    filteredMatches = []
    #print(len(knnmatchesLtoR),"   -   ",len(knnmatchesRtoL))
    for i in range(len(knnmatchesLtoR)):
       if(knnmatchesLtoR[i][0].queryIdx == knnmatchesRtoL[i][1].trainIdx):
            filteredMatches.append(knnmatchesLtoR[i][0])
   
    
    return filteredMatches


def ratioTest(knnmatches, ratioThreshold):
    """
    ratio test is performed for the given matches for the provided threshold

    Parameters
    ----------
    knnmatches : TYPE
        DESCRIPTION.
    ratioThreshold : TYPE
        DESCRIPTION.

    Returns
    -------
    filteredMatches : TYPE
        DESCRIPTION.

    """
    filteredMatches = []
    for distances in knnmatches:
        if (distances[0].distance) < ratioThreshold*distances[1].distance:
            filteredMatches.append(distances[0])
    return filteredMatches


def computeMatches(leftDesc, rightDesc):
    """
    KNN matches are found using the image descriptors

    Parameters
    ----------
    leftDesc : TYPE
        DESCRIPTION.
    rightDesc : TYPE
        DESCRIPTION.

    Returns
    -------
    filteredMatches : TYPE
        DESCRIPTION.

    """
    knnmatchesLtoR = matchKNN(leftDesc, rightDesc)
    #knnmatchesRtoL = matchKNN(rightDesc, leftDesc)
    #crossCheck(knnmatchesLtoR, knnmatchesRtoL)
    filteredMatches = ratioTest(knnmatchesLtoR, 0.75)

    return filteredMatches


def findHomographyMatrixFor4RandomPoints(pointSet1, pointSet2):
    """
    Normalised Homography matrix is calcualtated for given set of 4 points 

    Parameters
    ----------
    pointSet1 : TYPE
        DESCRIPTION.
    pointSet2 : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.

    """
    # 2(4)x9
    A = np.zeros((8, 9))
    for i in range(len(pointSet1)):
       
        row1 = [pointSet1[i][0], pointSet1[i][1], 1, 0, 0, 0, -pointSet2[i][0]
                * pointSet1[i][0], -pointSet2[i][0]*pointSet1[i][1], -pointSet2[i][0]]
        row2 = [0, 0, 0, pointSet1[i][0], pointSet1[i][1], 1, -pointSet2[i][1]
                * pointSet1[i][0], -pointSet2[i][1]*pointSet1[i][1], -pointSet2[i][1]]
        A[2*i] = np.array(row1)
        A[(2*i) + 1] = np.array(row2)
        #A[i*2:i*2 + 2] = np.array([row1,row2])

    # SVD decomposition on A
    U, s, vt = np.linalg.svd(A, full_matrices=True)
    # homogeneous solution of Ah=0 as the rightmost column vector of V.
    x = vt[-1]
    H = x.reshape(3, 3)
    H = H/H[2][2]
    return H

def computeHomography(leftKp, rightKp, matches, iterations):
    """
    Easier version of RANSAC is implemented.
    best fitting Inliers and Homography matrix is calcualted by randomly 
    finding Homography matrix for 4 points at random for given number of iterations

    Parameters
    ----------
    leftKp : TYPE
        DESCRIPTION.
    rightKp : TYPE
        DESCRIPTION.
    matches : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.

    Returns
    -------
    bestHomography : TYPE
        DESCRIPTION.

    """
    points1 = []
    points2 = []

    for i in range(len(matches)):  # DMatches
        points1.append(leftKp[matches[i].queryIdx].pt)
        points2.append(rightKp[matches[i].trainIdx].pt)

    bestInliers = 0
    bestHomography = None
    for i in range(iterations):
        subset1 = []
        subset2 = []
        # Construct the subsets by randomly choosing 4 matches.
        for x in range(4):
            randIndex = random.randrange( len(points1) - 1)
            subset1.append(points1[randIndex])
            subset2.append(points2[randIndex])
        # Compute the homography for this subset
        H = findHomographyMatrixFor4RandomPoints(subset1, subset2)
        # Compute the number of inliers
        inliers = 0
        points1Mat3x3 = np.vstack(
            (np.array(points1).T, np.ones((1, np.array(points1).T.shape[1]))))
        points2Mat3x3 = np.vstack(
            (np.array(points2).T, np.ones((1, np.array(points2).T.shape[1]))))
        H3x3 = np.matmul(H, points1Mat3x3)
        xdivided = H3x3/H3x3[-1, :]
        diffSquare = (xdivided - points2Mat3x3)**2
        distance = np.sqrt(np.sum(diffSquare, axis=0))  # traverse vertical
        for i in range(len(distance)):
            if distance[i] < 5:
                inliers = inliers+1
        
        # Keep track of the best homography and inliers 
        if inliers > bestInliers:
            bestInliers = inliers
            bestHomography = H
    return bestHomography

def createStichedImage(H, left_img, right_img):
    """
    Generating panaroma where left image is warped to fit into right image
    using perspectiveTransform and warpPerspective

    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    left_img : TYPE
        DESCRIPTION.
    right_img : TYPE
        DESCRIPTION.

    Returns
    -------
    panaromaImage : TYPE
        DESCRIPTION.

    """

    leftImgHeight, leftImgWidth, leftImgChannel = left_img.shape
    rightImgHeight, rightImgWidth, rightImgChannel = right_img.shape

    minX = 0
    minY = 0
    maxX = rightImgWidth
    maxY = rightImgHeight

    corners = [[0, 0], [leftImgWidth, 0], [
        leftImgWidth, leftImgHeight], [0, leftImgHeight]]
    for i in range(len(corners)):
        maxX = max(maxX, corners[i][0])
        minX = min(minX, corners[i][0])
        maxY = max(maxY, corners[i][1])
        minY = min(minY, corners[i][1])

    corners = np.array(corners, dtype=float)
    corners = cv2.perspectiveTransform(
        corners[None, :, :], H)

    for corner in corners[0]:
        minX = min(minX, corner[0])
        maxX = max(maxX, corner[0])
        minY = min(minY, corner[1])
        maxY = max(maxY, corner[1])

    panaromaImageMatrix = np.array([np.floor(minX), np.floor(minY), np.ceil(
        maxX) - np.floor(minX), np.ceil(maxY) - np.floor(minY)]).astype(int)
      
    leftImgH = H
    offsetX = np.floor(minX)
    offsetY = np.floor(minY)

    # change Right image's H value with offset of panaroma
    rightImgH = np.eye(3)
    rightImgH[0, 2] = -1*offsetX
    rightImgH[1, 2] = -1*offsetY
    # change Left image's H value wrt the rights new offset
    leftImgH = np.matmul(rightImgH, leftImgH)
    panaromaImage = np.zeros([panaromaImageMatrix[3], panaromaImageMatrix[2], 3], dtype=int)
    # add right image 
    warpedRightImg = cv2.warpPerspective(cv2.cvtColor(cv2.cvtColor(
        right_img, cv2.COLOR_RGB2RGBA), cv2.COLOR_RGB2RGBA), rightImgH, (panaromaImageMatrix[2], panaromaImageMatrix[3]))
    for y in range(panaromaImage.shape[0]):
        for x in range(panaromaImage.shape[1]):
            if (warpedRightImg[y, x, 3] == 255):
                panaromaImage[y, x] = warpedRightImg[y, x, 0:3]
    # Added left imgae to to the sitched matrix
    warpedLeftImg = cv2.warpPerspective(cv2.cvtColor(
        left_img, cv2.COLOR_RGB2RGBA), leftImgH, (panaromaImageMatrix[2], panaromaImageMatrix[3]))
    for y in range(panaromaImage.shape[0]):
        for x in range(panaromaImage.shape[1]):
           # avoid edge non dark borders and already stiched data
            if (warpedLeftImg[y, x, 3] == 255 and np.array_equal(panaromaImage[y, x], np.array([0, 0, 0]))):
                panaromaImage[y, x] = warpedLeftImg[y, x, 0:3]
    return panaromaImage


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    #raise NotImplementedError

    print("Starting KeyPoint and Descriptor extraction ")
    sift = cv2.xfeatures2d.SIFT_create(1000) 
    leftKps, leftDesc = sift.detectAndCompute(left_img, None)
    rightKps, rightDesc = sift.detectAndCompute(right_img, None)

    print("Starting Computing keypoint matches")
    matches = computeMatches(leftDesc, rightDesc)

    print("Starting Homography matrix estimation")
    H = computeHomography(leftKps, rightKps, matches, 5000)

    print("Starting Warp and Image stitching ")
    result_img = createStichedImage(H, left_img, right_img)

    print("Panaroma generated ")
    return result_img


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
