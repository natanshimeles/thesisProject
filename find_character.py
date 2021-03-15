import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from Image_Preprocessor import *
class Segmentation:
    def __init__(self,file_path):
        print("\n........Program Initiated.......\n")
        self.src_img= cv2.imread(file_path, 1)
        copy = self.src_img.copy()
        height = self.src_img.shape[0]
        width = self.src_img.shape[1]
        print("\n Resizing Image........")
        self.src_img = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)

        height = self.src_img.shape[0]
        width = self.src_img.shape[1]

        print("#---------Image Info:--------#")
        print("\tHeight =",height,"\n\tWidth =",width)
        print("#----------------------------#")

        grey_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        print("Applying Adaptive Threshold with kernel :- 21 X 21")
        self.bin_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
        self.bin_img1 = self.bin_img.copy()
        self.bin_img2 = self.bin_img.copy()

    def close_openings(self):
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
            # final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
            # final_thr = cv2.dilate(bin_img,kernel1,iterations = 1)
            print("Noise Removal From Image.........")
            self.final_thr = cv2.morphologyEx(self.bin_img, cv2.MORPH_CLOSE, self.kernel)
            self.contr_retrival = self.final_thr.copy()



    def findContours(self):
        contr_img, contours, hierarchy = cv2.findContours(self.contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.final_contr = np.zeros((self.final_thr.shape[0],self.final_thr.shape[1],3), dtype = np.uint8)
        #cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

        i = 0
        c = None
        im_pre = ImagePreprocessor()


        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x,y,w,h = cv2.boundingRect(cnt)

                new_img = self.src_img[y:y+h, x:x+w]


                #new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, self.kernel)


                new_img =im_pre.noise_removal(new_img)
                new_img =im_pre.ImageScaling(new_img)
                new_img =im_pre.reverse_image(new_img)

                cv2.imwrite("segmented_characters/" + str(i) + ".jpg", new_img)

                if c is None:
                    c = np.array([new_img])
                else:
                    new_img = np.array([new_img])
                    c= np.append(c,new_img,axis=0)
                i= i+1
        #-------------/Thresholding Image-------------#
        return c




if __name__ == '__main__':
    s = Segmentation()
    s.close_openings()
    c = s.findContours()
