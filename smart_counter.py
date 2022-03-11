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

    return BLUR, minVal, maxVal, DILATE_SIZE, MIN_CONT_THRESHOLD, Color_Settings

def color_limits(C_S):

    """ The upper and lower bounds of the color limits are set in the config
        file, howerver these are stored as a string and need to be converted
        to an array to be used by the program, this is done in this function
    """
    RED_U = np.array([int(C_S["RBU"]), int(C_S["RGU"]), int(C_S["RRU"])])
    RED_L = np.array([int(C_S["RBL"]), int(C_S["RGL"]), int(C_S["RRL"])])
    BLUE_U = np.array([int(C_S["BBU"]), int(C_S["BGU"]), int(C_S["BRU"])])
    BLUE_L = np.array([int(C_S["BBL"]), int(C_S["BGL"]), int(C_S["BRL"])])
    GREEN_U = np.array([int(C_S["GBU"]), int(C_S["GGU"]), int(C_S["GRU"])])
    GREEN_L = np.array([int(C_S["GBL"]), int(C_S["GGL"]), int(C_S["GRL"])])

    return RED_L, RED_U, GREEN_L, GREEN_U, BLUE_L, BLUE_U

def filter_contours(contours, min):
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
    total_smarties = len(filtd_contours)

    return filtd_contours, filtered_conatours_df

def s_color_counter(smty_df, img):

    # 1) find the centroid of each smartie
    cont = smty_df["Contour"].to_list()
    centroid = []
    for i in cont:
        x_val = []
        y_val = []
        for j in i:
            x_val.append(j[0][0])
            y_val.append(j[0][1])
        centroid.append([statistics.mean(x_val), statistics.mean(y_val)])
        # this_centroid = np.array([centroid[-1]])
        # print((this_centroid))
        # cv.drawContours(img, this_centroid, -1, (255,255,255), 20)
        # plt.imshow(img)
        # plt.show()


    print("Ammount of Smarties")
    print(len(centroid))

    centroid_contours = []
    for i in centroid:
        # this specific format is requied for the drawContours function
        centroid_contours.append(np.array([i]))

    # print(centroid_contours)

    smty_df["Centroid"] = centroid

    # s_color_val = []
    # for i in centroid:
    #     s_color_val.append(img[i])

    # print(s_color_val)


    cv.drawContours(img, centroid_contours, -1, (255,255,255), 20)
    plt.imshow(img)
    plt.show()

def main(blr_amt, minVal, maxVal, dilate_size, MIN_CONT_THRESHOLD):

    # 1) Load up the image file
    # opencv uses BGR when it imports images
    og_img_file_loc = get_dir_path("Red_15.JPG")

    # og_img_file_loc = get_dir_path("Smarties.jpg")
    # og_img_file_loc = get_dir_path("Smartie_Dark_2.JPG")
    og_img = cv.imread(og_img_file_loc) # BGR
    og_img1 = cv.imread(og_img_file_loc) # BGR (extra image for later use)

    # 2) convert to HSV
    col_conv_img = cv.cvtColor(og_img, cv.COLOR_BGR2RGB)
    plt.imshow(col_conv_img)
    plt.show()

    # 3) Add blur to image
    blur_img = cv.GaussianBlur(col_conv_img, (blr_amt, blr_amt), cv.BORDER_DEFAULT)
    plt.imshow(blur_img, cmap='gray')
    plt.show()

    # 4) Find the edges using canny
    canny_img = cv.Canny(blur_img, minVal, maxVal)
    plt.imshow(canny_img, cmap='gray')
    plt.show()

    kernel = np.ones((dilate_size, dilate_size))

    dilate_img = cv.dilate(canny_img, kernel, iterations=1)
    plt.imshow(dilate_img, cmap='gray')
    plt.show()

    # 5) Find the contours in the image
    contours, heirarchy = cv.findContours(dilate_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # 6) Filter out all the small contours (these are usually errors)

    filtd_contours, filtered_conatours_df = filter_contours(contours, MIN_CONT_THRESHOLD)
    # print(filtd_contours)
    # for i in contours:
    #     cv.drawContours(og_img, i, -1, (200,200,200), 14)
    #     plt.imshow(og_img)
    #     plt.show()

    cv.drawContours(og_img, contours, -1, (200,200,200), 14)
    plt.imshow(og_img)
    plt.show()

    # 7) count the number of different smartie colors:

    smatie_colors = s_color_counter(filtered_conatours_df, og_img)




if __name__ == '__main__':
    blr_amt, minVal, maxVal, dilate_size, MIN_CONT_THRESHOLD, Color_Settings = get_settings()
    RED_L, RED_U, GREEN_L, GREEN_U, BLUE_L, BLUE_U = color_limits(Color_Settings)
    main(blr_amt, minVal, maxVal, dilate_size, MIN_CONT_THRESHOLD)
