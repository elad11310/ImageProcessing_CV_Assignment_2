from ex2_utils import *
import matplotlib.pyplot as plt
import cv2


def main():
    #
    # # print("hello")
    # a = [4]
    # # b = [1,2,3,4,5,73,2,5,3]
    # #
    # # print(np.convolve(b,a,"full"))
    # # a  = conv1D(b,a) ## if we send it a,b it works because convolution is commutative
    # # print(a)
    img_path = 'coins.jpg'
    #
    # img = cv2.imread(img_path, 0)
    # img = cv2.medianBlur(img, 5)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
    #                           param1=50, param2=30, minRadius=70, maxRadius=73)
    #
    # for i in circles[0, :]:
    #     ##circles = np.uint16(np.around(circles))
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow('detected circles', cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ## open in grey
    ## [(92, 20, 18), (33, 30, 15), (135, 43, 18), (67, 52, 15), (165, 63, 15), (108, 74, 18), (147, 107, 15), (75, 129, 18)]
    o = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    cv2.circle(o, (75, 129), 18, (0, 255, 0), 2)
    cv2.imshow("Detected Circle", o)
    cv2.waitKey(0)
    m = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # # ## for gaussian filter
    e = [[1, 2, 1, 1, 1], [2, 4, 2, 1, 1], [1, 2, 1, 2, 3], [1, 2, 1, 3, 4], [1, 2, 3, 4, 5]]
    # # ## for avg filter
    # f = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # img3= np.zeros_like(m)
    # # ## for median filter  - best for sale and pepper noise.
    # g = cv2.medianBlur(m,3,img3)
    #
    # ## making the list as np.array
    # ## k = np.array(f,float)
    # d = np.array(e, float)
    # img = np.zeros_like(m)
    # img=cv2.filter2D(m,2,d)
    # # img2 = np.zeros_like(m)
    # # img2 = cv2.filter2D(m,-1,d)
    # #
    # img4 = np.zeros_like(m)
    # img4 = conv2D(m,d)
    #
    # plt.gray()
    # f,ax = plt.subplots(1,2)
    # ax[0].imshow(img)
    # ax[1].imshow(img4)
    # plt.show()
    ##a, b, c, d = convDerivative(m)
    ## blurImage1(m,63)
    ##blurImage2(m, 50)
    ##img2 = edgeDetectionZeroCrossingLOG(m)
    # imgTest = cv2.Laplacian(m,cv2.CV_64F)
    ##imggg = edgeDetectionSobel(m)
    ## imgggg = cv2.Sobel(m,cv2.CV_64F,0,2)
    ##plt.gray()
    ##f, ax = plt.subplots(1, 2)
    ##ax[0].imshow(b)
    ##ax[1].imshow(imggg)
    ##  ax[2].imshow(img2)
    ##plt.show()
    # im1,im2 = edgeDetectionCanny(m,0.05,0.09)
    # plt.gray()
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(im1)
    # ax[1].imshow(im2)
    # plt.show()
    ##houghCircle(m, 13, 20)


# d = [[1,2,3],[6,7,8],[9,8,6]]
# # # print(d)
# #
# #
# h = np.array(d)
# print(h)
#
# ##h=np.repeat(h, [4, 1, 1], axis=0)
# h = np.pad(h, pad_width=1, mode='symmetric')
# ##h= np.repeat(h,6,axis=0)
# print(h)
#
# print("------------")
# h=h[1:-1,1:-1]
# print(h)


if __name__ == '__main__':
    main()
