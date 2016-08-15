"""This file is where the subjective parameters are set"""


def path():

    # 'path to extracted images'  # This should be the path to the folder containing the images
    extracted_images = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/test'

    return extracted_images


def save():

    # 'path to save the images'  # This should be the path where you would like images to be saved
    save_path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/save_images/segment1/'

    return save_path


def params():

    minimum_nest_size = 100  # This is the minimum nest size in pixels
    minimum_hole_size = 500  # This is the minimum hole size in pixels

    return minimum_nest_size, minimum_hole_size


def csv_save():

    # 'path to the csv file which contains the output data'
    csv_path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9'

    return csv_path