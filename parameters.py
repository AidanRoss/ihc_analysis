"""This file is where the subjective parameters are set"""


def path():

    # 'path to extracted images'  # This should be the path to the folder containing the images
    extracted_images = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/test'
    # extracted_images = '/Users/engs1348/Raquel/Nottingham-TMAs/tma-extracted'  # Raquel Path
    return extracted_images


def save():

    # 'path to save the images'  # This should be the path where you would like images to be saved
    save_path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/save_images/segment2/'
    # save_path = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan' # Raquel Path

    return save_path


def params():

    minimum_nest_size = 100  # This is the minimum nest size in pixels
    minimum_hole_size = 500  # This is the minimum hole size in pixels

    return minimum_nest_size, minimum_hole_size


def microns():

    # This value should be the pixel resolution of the images being analyzed
    # If the images are extracted at the same resolution as the scanner then the pixel res is 0.0456 mpp-x
    pixel_res = 0.0456  # microns per pixel_x
    pixel_size = pixel_res**2

    return pixel_res, pixel_size


def csv_save():

    # 'path to the csv file which contains the output data'
    csv_path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9'
    micron_csv = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9'

    # csv_path = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan'    # Raquel Path
    # micron_csv = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan'  # Raquel Path to new csv

    return csv_path, micron_csv
