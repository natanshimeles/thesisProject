import cv2 as cv
import numpy as n
from matplotlib import pyplot as plot
from skimage.io import imread
from skimage.filters import threshold_otsu


class ImagePreprocessor:
    def ImageScaling(self,image):
        #oriimg = cv.imread("die.jpg",0)
        newX,newY = 32,32
        newimg = cv.resize(image,(int(newX),int(newY)))
        #cv.imwrite("out.jpg",dst)
        return newimg

    def noise_removal(self,image):
        dst = cv.fastNlMeansDenoisingColored(image,None,10,10,7,21)
        #cv.imwrite("out.jpg",dst)
        return dst
    def reverse_image(self,image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = (255-image)
        return image

    def changetobinary(self,image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        im_bw =reverse_image(im_bw)
        im_bw = im_bw/255


        return im_bw
