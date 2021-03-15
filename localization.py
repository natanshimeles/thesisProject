from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

class Localization:
    def __init__(self,dir):
        self.dir = dir

    def tobinary(self):
        car_image = imread(self.dir, as_gray=True)
        gray_car_image = car_image * 255
        threshold_value = threshold_otsu(gray_car_image)
        self.binary_car_image = gray_car_image > threshold_value
        
        return self.binary_car_image
