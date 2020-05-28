import cv2
import numpy as np
import math


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203306014


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    ## first check size of the kernel - if odd take the middle (size/2), if even take middle-1 (size/2-1).
    ##
    ## checking size of the kernel
    sizeKernel = len(kernel1)
    ## we are gonna use "full" so size of h : size(A) + Size(B) -1
    h = np.zeros(len(inSignal) + sizeKernel - 1, int)
    ## if the size is even
    if sizeKernel % 2 == 0:
        mid = (int)((sizeKernel / 2) - 1) - 1

    else:
        mid = (int)(sizeKernel / 2) - 1

    ## for the sigma
    start = -mid
    end = mid + 2

    if mid <= 0:
        start = 0
        end = sizeKernel - 1
        mid = 0

    ## now adding zeros before and after the original arr as the size of the kernel.
    newArr = np.zeros(len(inSignal) + sizeKernel * 2)
    k = 0
    for i in range(sizeKernel, sizeKernel + len(inSignal)):
        newArr[i] = inSignal[k]
        k += 1

    for x in range(0, len(h)):
        for i in range(start, end + 1):
            h[x] += kernel1[mid + i] * newArr[sizeKernel - (mid + i)]
        sizeKernel += 1

    return h


def flipKernel(kernel: np.ndarray):
    h, w = kernel.shape
    img_copy = np.zeros_like(kernel, int)

    for i in range(h):
        for j in range(w):
            img_copy[i][j] = kernel[h - i - 1][w - j - 1]

    return img_copy


def conv2D(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    return output


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    #  [1,0,−1]T , [1,0,-1] to both kernel we need to flip them (will happen in the convolution function)
    # Iy - using kernel [1,0,-1] in order to find edges in the Y directions
    # Ix - using kernel [1,0,-1]T in order to find edges in the X directions
    # to find the magnitude- for each pixel -> power it up-> sum them(x+y) -> and the magnitude will be the square root of the sum
    # to find the direction - for each pixel -> divide(y/x) -> on the result we will perform tan-1 and it will be the new value
    k = [[1, 0, -1]]
    kernel = np.array(k, float)
    kernelT = np.transpose(kernel)

    # getting the edges in each direction
    x_der = conv2D(inImage, kernelT)
    y_der = conv2D(inImage, kernel)

    magnitude = np.sqrt((x_der ** 2) + (y_der ** 2))
    directions = np.arctan2(y_der, x_der)
    return directions, magnitude, x_der, y_der


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    # in order to get a blurred image, we need to take a gaussian filter and do convolution on the image with that filter.
    kernel = cv2.getGaussianKernel(kernel_size, math.sqrt(kernel_size))  # getting filter from cv2
    img_new = conv2D(in_image, kernel)
    return img_new


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    kernel = cv2.getGaussianKernel(kernel_size, math.sqrt(kernel_size))  # getting filter from cv2
    img = cv2.filter2D(in_image, 2,
                       kernel)  # sending it to convolution function using cv2 as requested to use cv2 built in functions
    return img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    kernel_size = 3
    ##first blurring using gaussian filter.
    img = blurImage2(img, kernel_size)
    ## making laplacian kernel
    ## we are using this specific filter because its doing the second derivetive for axis X and axis Y simultaneously
    ## becuse for axis x we use [1,-2,1] and for axis y we use [1,-2,1]T so
    ## this kernel is doing the job once for axis x and axis y.
    LaplacianFilter = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    Laplaciankernel = np.array(LaplacianFilter, float)
    ## making convolution with the laplacian kernel
    img = conv2D(img, Laplaciankernel)
    # plt.gray()
    # plt.imshow(img)
    # plt.show()
    return img


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7):
    axisXKernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
    axisYKernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)

    img = img.dot(1 / 255)
    img_x = conv2D(img, axisXKernel)
    img_y = conv2D(img, axisYKernel)

    img_magnitude = np.sqrt((img_x ** 2) + (img_y ** 2))

    img_magnitude[img_magnitude <= thresh] = 0
    img_magnitude[img_magnitude > thresh] = 1

    return img_magnitude


def edgeDetectionSobel2(img: np.ndarray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = conv2D(img, Kx)
    Iy = conv2D(img, Ky)

    img_magnitude = np.hypot(Ix, Iy)  ## equivalnet to  np.sqrt((Ix ** 2) + (Iy ** 2))
    img_magnitude = img_magnitude / img_magnitude.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (img_magnitude, theta)


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    ## step 1 - blurring with gaussian
    img1 = blurImage2(img, 3)

    ## finding second derivetive
    ##The Gradient calculation step detects the edge intensity and direction
    ## by calculating the gradient of the image using edge detection operators.
    img1, Gradient = edgeDetectionSobel2(img1)

    ## step 3 non max suppression
    img1 = non_max_suppression(img1, Gradient)

    ## step 4 double treshhold - decdeing on 2 treshholds 1 high and 1 low
    ## Strong pixels are pixels that have an intensity so high that we are sure its an edge
    # Weak pixels are pixels that have an intensity value that is not enough to be considered as strong ones,
    # but yet not small enough to be considered as non-relevant for the edge detection.(means larger the the low treshhold and smaller
    ##then the high treshold) we will take care of them in the next step.
    ##Other pixels are considered as non-relevant for the edge.means they are not edge for sure.
    res, strong, weak = threshold(img1, thrs_1, thrs_2)  ## goond tresholds : 0.05 , 0.09

    ## step 5 - taking care of the pixels among the weak and strong. we will check every weak pixel's neighbours
    ## if any of them is strong we gonna change the intensity of the proccesd one to strong , otherwise to 0.

    res = hysteresis(res, weak, strong)

    return res, cv2.Canny(img, thrs_1, thrs_2, 3)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio, highThresholdRatio):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)
    ## saving all the indices where we have pixels in the img greater then the current highThreshold.
    strong_i, strong_j = np.where(img >= highThreshold)
    ##saving all the indices where we have pixels in the img smaller then the current lowThreshold.
    ##zeros_i, zeros_j = np.where(img < lowThreshold)

    ##saving all the indices where we have pixels in the img greater then the current lowThreshold,and smaller then
    ## the current highThreshold (amongs them).
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, strong, weak


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    ## first step - Initializing the Accumulator Matrix: Initialize a matrix of dimensions rows * cols * maxRadius with zeros.
    Row, Col = img.shape
    A = np.zeros((Row, Col, max_radius))
    ## step two - Apply blurring, grayscale and an edge detector on the image.
    # This is done to ensure the circles show as darkened image edges.
    Canny, Canny2 = edgeDetectionCanny(img, 0.05, 0.09)

    ##step 3 - Vote the all possible circles in accumulator.
    for i in range(0, Row):
        for j in range(0, Col):
            if (Canny[i][j] == 255):
                for r in range(min_radius, max_radius):
                    for teta in range(0, 360):
                        a = i - r * math.cos(teta * math.pi / 180)
                        b = j - r * math.sin(teta * math.pi / 180)
                        if a >= Row or b >= Col:
                            continue
                        A[int(a)][int(b)][r] += 1


## finding the max value in the 3D Accumulator
    list = []
    max = 0
    for i in range(Row):
        for j in range(Col):
            for k in range(min_radius, max_radius):
                if A[i][j][k] != 0:
                    if A[i][j][k] > max:
                        max = A[i][j][k]




    ## finding potential center of circles.
    for i in range(Row):
        for j in range(Col):
            for k in range(min_radius, max_radius):
                if A[i][j][k] > max - 30 and A[i][j][k] <= max:
                    list.append((j, i, k))


    ##  to strain similar circles.
    treshOld = 10
    i=0
    while(i<len(list)):
        a, b, c = list[i]
        if i != len(list) - 1:
           j=i+1
           while(j<len(list)):
                e, f, g = list[j]
                if a - treshOld <= e <= a + treshOld and b - treshOld <= f <= b + treshOld:
                    list.remove(list[j])
                    j-=1
                j+=1
        i+=1


    return list
