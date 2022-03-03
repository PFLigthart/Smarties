import cv2 as cv
import numpy as np
import os
import configparser
import matplotlib.pyplot as plt


def get_dir_path(target_filename):

    """
    Determine the absolute path to a file. This path manipulation is
    here so that the script will run successfully regardless of the
    current working directory.
    """

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, target_filename)

    return config_path

def get_settings():

    # create settings object from config file
    settings_obj = configparser.ConfigParser()
    settings_obj.read(get_dir_path("smartie_config.ini"))
    Settings = settings_obj["Settings"]
    BLUR = int(Settings["BLUR"])
    minVal = int(Settings["MINVAL"])
    maxVal = int(Settings["MAXVAL"])
    DILATE_SIZE = int(Settings["DILATE_SIZE"])

    return BLUR, minVal, maxVal, DILATE_SIZE

def main(blr_amt, minVal, maxVal, dilate_size):

    # 1) Load up the image file
    og_img_file_loc = get_dir_path("Smarties.jpg")
    og_img = cv.imread(og_img_file_loc) # BGR

    # 2) convert to HSV
    hsv_img = cv.cvtColor(og_img, cv.COLOR_RGB2HSV)
    plt.imshow(hsv_img)
    plt.show()

    # 3) Add blur to image
    blur_img = cv.GaussianBlur(hsv_img, (blr_amt, blr_amt), cv.BORDER_DEFAULT)
    # plt.imshow(blur_img, cmap='gray')
    # plt.show()

    # 4) Find the edges using canny
    canny_img = cv.Canny(blur_img, minVal, maxVal)
    # plt.imshow(canny_img, cmap='gray')
    # plt.show()

    kernel = np.ones((dilate_size, dilate_size))

    dilate_img = cv.dilate(canny_img, kernel, iterations=1)
    plt.imshow(dilate_img, cmap='gray')
    plt.show()

if __name__ == '__main__':

    BLUR, minVal, maxVal, DILATE_SIZE = get_settings()
    main(BLUR, minVal, maxVal, DILATE_SIZE)
