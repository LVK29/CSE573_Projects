"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used
image processing techniques: image denoising and edge detection.
Specifically, you are given a grayscale image with salt-and-pepper noise,
which is named 'task2.png' for your code testing.
Note that different image might be used when grading your code.

You are required to write programs to:
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint:
• Zero-padding is needed before filtering or convolution.
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2.
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation.
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering.
Please write the convolution code ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here
    #raise NotImplementedError

    result = np.zeros(shape=(len(img), len(img[0])))

    img = np.pad(img, pad_width=1, mode='constant', constant_values=0)


    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):
            # logic to iterate over 3x3
            tct = []
            tct.append(img[i-1][j-1])
            tct.append(img[i-1][j])
            tct.append(img[i-1][j+1])
            tct.append(img[i][j-1])
            tct.append(img[i][j])
            tct.append(img[i][j+1])
            tct.append(img[i+1][j-1])
            tct.append(img[i+1][j])
            tct.append(img[i+1][j+1])
            result[i-1][j-1] = np.median(tct)
    return result


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    # flip kernel
    kernel = np.flip(kernel)
    img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    result = np.zeros(shape=(len(img), len(img[0])))
    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):
            sum = 0
            for k in range(len(kernel)):
                for l in range(len(kernel[0])):
                    sum = sum + kernel[k][l]*img[i+k-1][j+l-1]
            result[i-1][j-1] = sum
    list = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] < 0):
                result[i][j] = result[i][j] * -1
            list.append(result[i][j])
    return result


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image,
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    edge_x = convolve2d(img, sobel_x)
    edge_y = convolve2d(img, sobel_y)
    edge_mag = np.zeros(shape=(len(edge_x), len(edge_x[0])))
    list = []
    for i in range(len(edge_x)):
        for j in range(len(edge_x[0])):
            edge_mag[i][j] = (edge_x[i][j]**2 + edge_y[i][j]**2)**(1/2)
            list.append(edge_mag[i][j])
    edge_mag = edge_mag / max(list)
    # normalise
    edge_x = 255 * (edge_x - np.min(edge_x)) / (np.max(edge_x) - np.min(edge_x))
    edge_y = 255 * (edge_y - np.min(edge_y)) / (np.max(edge_y) - np.min(edge_y))
    edge_mag = 255 * (edge_mag - np.min(edge_mag)) / (np.max(edge_mag) - np.min(edge_mag))

    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    sobel_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).astype(int)
    sobel_45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(int)
    edge_45 = convolve2d(img, sobel_45)
    edge_135 = convolve2d(img, sobel_135)
    # normalise
    edge_45 = 255 * (edge_45 - np.min(edge_45)) / (np.max(edge_45) - np.min(edge_45))
    edge_135 = 255 * (edge_135 - np.min(edge_135)) / (np.max(edge_135) - np.min(edge_135))

    # print() # print the two kernels you designed here
    print(sobel_45)
    print(sobel_135)
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)
