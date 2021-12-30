"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np



#structureElement = np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(int)
def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    result = np.full((len(img)-2 , len(img[0])-2), False, dtype=bool)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            if img[i-1][j-1]==True and img[i-1][j]==True and img[i-1][j+1]==True and img[i][j-1]==True and img[i][j]==True and img[i][j+1]==True and img[i+1][j-1]==True and img[i+1][j]==True and img[i+1][j+1]==True:
                result[i-1][j-1]=True
    return result
   

def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    
    result = np.full((len(img)-2 , len(img[0])-2 ), False, dtype=bool)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            if img[i-1][j-1]==True or img[i-1][j]==True or img[i-1][j+1]==True or img[i][j-1]==True or img[i][j]==True or img[i][j+1]==True or img[i+1][j-1]==True or img[i+1][j]==True or img[i+1][j+1]==True:
                result[i-1][j-1]=True
    return result


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img = np.pad(img, pad_width=1, mode='constant', constant_values=False)
    erodedImg = morph_erode(img)
    erodedImg = np.pad(erodedImg, pad_width=1, mode='constant', constant_values=False)
    dilatedImg = morph_dilate(erodedImg)
    return dilatedImg


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img = np.pad(img, pad_width=1, mode='constant', constant_values=False)
    dilatedImg = morph_dilate(img)
    dilatedImg = np.pad(dilatedImg, pad_width=1, mode='constant', constant_values=False)
    erodedImg = morph_erode(dilatedImg)
   
    return erodedImg
    #return close_img


def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img_binary = np.full((len(img) , len(img[0]) ), False, dtype=bool)
    result = np.full((len(img) , len(img[0]) ), 0, dtype=int)
    
    for i in range(len(img)):
       for j in range(len(img[0])):
           if img[i][j]==0:
               img_binary[i][j]=False
           else:
               img_binary[i][j]=True
               
               
    temp = morph_open(img_binary)
    res = morph_close(temp)
    
    for i in range(len(res)):
       for j in range(len(res[0])):
           if res[i][j]==True:
               result[i][j]=255
           else:
               result[i][j]=0
               
    return result


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    img_binary = np.full((len(img) , len(img[0]) ), False, dtype=bool)
    result = np.full((len(img) , len(img[0]) ), 0, dtype=int)
    
    for i in range(len(img)):
       for j in range(len(img[0])):
           if img[i][j]==0:
               img_binary[i][j]=False
           else:
               img_binary[i][j]=True
               
    img_binary = np.pad(img_binary, pad_width=1, mode='constant', constant_values=False)
    erodedImg = morph_erode(img_binary)   
    
    
    
    
    for i in range(len(erodedImg)):
       for j in range(len(erodedImg[0])):
           if erodedImg[i][j]==True:
               result[i][j]=255
           else:
               result[i][j]=0
    boundary = np.subtract(img, result)            
    return boundary
               
         


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)





