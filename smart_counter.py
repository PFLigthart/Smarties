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
    BLUR = int(Settings["BLUR"])
    minVal = int(Settings["MINVAL"])
    maxVal = int(Settings["MAXVAL"])
    DILATE_SIZE = int(Settings["DILATE_SIZE"])
    MIN_CONT_THRESHOLD = float(Settings["MIN_CONT_THRESHOLD"])

    return BLUR, minVal, maxVal, DILATE_SIZE, MIN_CONT_THRESHOLD

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


    centroid_contours = []
    for i in centroid:
        # this specific format is requied for the drawContours function
        centroid_contours.append(np.array([i]))

    # centroid_np_array = np.array([centroid_contours])
    # print(centroid_np_array)


    smty_df["Centroid"] = centroid
    # print(smty_df)

    cv.drawContours(img, centroid_contours, -1, (255,250,250), 12)
    plt.imshow(img)
    plt.show()


def main(blr_amt, minVal, maxVal, dilate_size, MIN_CONT_THRESHOLD):

    # 1) Load up the image file
    og_img_file_loc = get_dir_path("Smarties.jpg")
    og_img = cv.imread(og_img_file_loc) # BGR
    og_img1 = cv.imread(og_img_file_loc) # BGR (extra image for later use)

    # 2) convert to HSV
    hsv_img = cv.cvtColor(og_img, cv.COLOR_RGB2HSV)
    # plt.imshow(hsv_img)
    # plt.show()

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
    # plt.imshow(dilate_img, cmap='gray')
    # plt.show()

    # 5) Find the contours in the image
    contours, heirarchy = cv.findContours(dilate_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # 6) Filter out all the small contours (these are usually errors)

    filtd_contours, filtered_conatours_df = filter_contours(contours, MIN_CONT_THRESHOLD)
    # print(filtd_contours)
    cv.drawContours(og_img, filtd_contours, -1, (255,255,240), 2)
    # plt.imshow(og_img)
    # plt.show()

    # 7) count the number of different smartie colors:

    smatie_colors = s_color_counter(filtered_conatours_df, og_img)




if __name__ == '__main__':
    blr_amt, minVal, maxVal, dilate_size, MIN_CONT_THRESHOLD = get_settings()
    main(blr_amt, minVal, maxVal, dilate_size, MIN_CONT_THRESHOLD)
