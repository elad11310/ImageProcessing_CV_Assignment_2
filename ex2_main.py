from ex2_utils import *
import cv2


def main():

   ## first ex - 1D convolution
    a = [1,0,1]
    b = [1,2,3,4,5,73,2,5,3]

    a  = conv1D(b,a) ## if we send it a,b it works because convolution is commutative

    img_path = 'coins.jpg'
    ## open in grey
    m = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # # ## for gaussian filter
    gaussian = [[1, 2, 1, 1, 1], [2, 4, 2, 1, 1], [1, 2, 1, 2, 3], [1, 2, 1, 3, 4], [1, 2, 3, 4, 5]]

    # ## making the list as np.array
    d = np.array(gaussian, float)

    ## second ex - 2d convolution
    img = conv2D(m,d)

    ## third ex - convolution by derivative
    a, b, c, d = convDerivative(m)
    ## forth ex - edge detection using LOG
    img = edgeDetectionZeroCrossingLOG(m)

    ## fifth ex - sobel edge detection
    img = edgeDetectionSobel(m)

    ## sixth ex - Canny edge detection
    im1,im2 = edgeDetectionCanny(m,0.05,0.09)

    list = houghCircle(m, 13, 20)

    print("success")

if __name__ == '__main__':
    main()
