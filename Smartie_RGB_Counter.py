import cv2 as cv
import numpy as np
import os
import configparser
import matplotlib.pyplot as plt
import statistics
import pandas as pd


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
    """ extracts settings from the configuration file """

    # create settings object from config file
    settings_obj = configparser.ConfigParser()
    settings_obj.read(get_dir_path("smartie_config.ini"))
    Settings = settings_obj["Settings"]
    Color_Settings = settings_obj["Color_Settings"]
    BLUR = int(Settings["BLUR"])
    minVal = int(Settings["MINVAL"])
    maxVal = int(Settings["MAXVAL"])
    DILATE_SIZE = int(Settings["DILATE_SIZE"])
    MIN_CONT_THRESHOLD = float(Settings["MIN_CONT_THRESHOLD"])

    return Color_Settings, BLUR, minVal, maxVal, DILATE_SIZE, MIN_CONT_THRESHOLD

def color_limits(C_S):

    """ The upper and lower bounds of the color limits are set in the config
        file, howerver these are stored as a string and need to be converted
        to an array to be used by the program, this is done in this function
    """
    RED_U = np.array([int(C_S["RBU"]), int(C_S["RGU"]), int(C_S["RRU"])], dtype = "uint8")
    RED_L = np.array([int(C_S["RBL"]), int(C_S["RGL"]), int(C_S["RRL"])], dtype = "uint8")
    BLUE_U = np.array([int(C_S["BBU"]), int(C_S["BGU"]), int(C_S["BRU"])], dtype = "uint8")
    BLUE_L = np.array([int(C_S["BBL"]), int(C_S["BGL"]), int(C_S["BRL"])], dtype = "uint8")
    GREEN_U = np.array([int(C_S["GBU"]), int(C_S["GGU"]), int(C_S["GRU"])], dtype = "uint8")
    GREEN_L = np.array([int(C_S["GBL"]), int(C_S["GGL"]), int(C_S["GRL"])], dtype = "uint8")

    Colors_List = [(RED_L, RED_U), (GREEN_L, GREEN_U), (BLUE_L, BLUE_U)]

    return Colors_List

def _filter_contours(contours, min):
    """ Filtering of the contours - to remove the small contours which are
        formed between correct contours as well as some errounious contours

        the min value supplied is the percentage of the mean contour value
        below which contours should be removed.
        i.e. if the mean contour is 100 then any conour with size less than
        100 * min will be remove. if min = 0.01, then a contour with size
        less than 10 will be excluded
    """

    cont_area = []
    for cont in contours:
        cont_area.append(cv.contourArea(cont))

    data = {"Contour": contours, "Area": cont_area}

    cont_df = pd.DataFrame(data)
    avg_contour = statistics.mean(cont_area)

    filtered_conatours_df = cont_df[cont_df["Area"] > (avg_contour*min)]

    filtd_contours = filtered_conatours_df["Contour"].to_list()
    return filtd_contours

def find_single_color_contour(img, minVal, maxVal, dilate_size, cont_thresh):

    """ Finds the contours in an image .incl filtering"""

        # Find the edges using canny
        canny_img = cv.Canny(img, minVal, maxVal)
        # plt.imshow(canny_img, cmap='gray')
        # plt.show()

        kernel = np.ones((dilate_size, dilate_size))

        # dilate the image for clearer edges
        dilate_img = cv.dilate(canny_img, kernel, iterations=1)
        # plt.imshow(dilate_img, cmap='gray')
        # plt.show()

        # Find the contours in the image
        contours, heirarchy = cv.findContours(dilate_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        actual_contours = _filter_contours(contours, cont_thresh)
        return actual_contours

def Count_a_color(clr, img, Colors_List, blr_amt, minVal, maxVal, dilate_size, cont_thresh):

    """ Counts the number of a single color in th given image """

    # the positions below refer to the positions of the lower and upper bounds
    # in the Colors_List array for each of the corresponding colors
    if clr == 'red':
        pos = 0
    elif clr == 'blue':
        pos = 2
    elif clr == 'green':
        pos = 1

    col_conv_img = img #cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(col_conv_img)
    # plt.show()

    # blur the image to deal with any unwanted minor differences
    blur_img = cv.GaussianBlur(col_conv_img, (blr_amt, blr_amt), cv.BORDER_DEFAULT)
    # plt.imshow(blur_img, cmap='gray')
    # plt.show()

    # Search for specific color

    color_mask = cv.inRange(blur_img, Colors_List[pos][0], Colors_List[pos][1])
    single_color_only = cv.bitwise_and(col_conv_img, col_conv_img, mask = color_mask)
    # plt.imshow(single_color_only)
    # plt.show()

    # find contours on the 'single color' image

    single_color_contours = find_single_color_contour(single_color_only, minVal, maxVal, dilate_size, cont_thresh)
    cv.drawContours(col_conv_img, single_color_contours, -1, (200,200,200), 14)
    plt.imshow(col_conv_img)
    plt.show()

    return len(single_color_contours)

def main(Colors_List, blr_amt, minVal, maxVal, dilate_size, cont_thresh):

    """ Main function that runs the over all code"""

    # 1) Load up the image file
    # opencv uses BGR when it imports images
    og_img_file_loc = get_dir_path("Final_img.JPG")

    og_img = cv.imread(og_img_file_loc) # BGR
    # og_img1 = cv.imread(og_img_file_loc) # BGR (extra image for later use)

    red_smarties = Count_a_color('red', og_img, Colors_List, blr_amt, minVal, maxVal, dilate_size, cont_thresh)
    blue_smarties  = Count_a_color('blue', og_img, Colors_List, blr_amt, minVal, maxVal, dilate_size, cont_thresh)
    green_smarties = Count_a_color('green', og_img, Colors_List, blr_amt, minVal, maxVal, dilate_size, cont_thresh)



if __name__ == '__main__':
    Color_Settings, blr_amt, minVal, maxVal, dilate_size, cont_thresh = get_settings()
    main(color_limits(Color_Settings), int(blr_amt), int(minVal), int(maxVal), int(dilate_size), float(cont_thresh))
