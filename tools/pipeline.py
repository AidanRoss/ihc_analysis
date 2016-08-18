""" This Script identifies useful nest properties for IHC images
Aidan Ross - TDI
@author: Aidan Ross
"""
import parameters
import csv
import os
import errno

import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# Import useful image analysis modules
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed, rgb2grey
from skimage.util import img_as_float, img_as_uint
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.measure import regionprops, ransac, CircleModel
from skimage.morphology import label, remove_small_objects, remove_small_holes
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max, canny
# from skimage.draw import circle_perimeter
# from skimage import io; io.use_plugin('matplotlib')

from scipy import ndimage as ndi

from operator import truediv
from math import pi


def color_conversion(img):
    # This function converts rgb image to the IHC color space, where channel 1 is Hematoxylin, 2 in Eosin and 3 is DAB

    ihc_rgb = skimage.io.imread(img)
    ihc_hed = rgb2hed(ihc_rgb)

    return ihc_rgb, ihc_hed


def rescale(img):
    # Rescaling the Image to a unsigned integer for Otsu thresholding method
    orig, ihc = color_conversion(img)
    rescaled = rescale_intensity(ihc[:, :, 2], out_range=(0, 1))  # rescaling channel 3 values between 0 and 1
    int_img = img_as_uint(rescaled)

    return int_img, orig, ihc


def create_bin(img):
    # Binary image created from Threshold, then labelling is done on this image

    int_img, orig, ihc = rescale(img)
    t_otsu = threshold_otsu(int_img)  # Global threshold using Otsu thresholding method - automatic
    bin_img = (int_img >= t_otsu)     # Creating binary image
    # bin =

    float_img = img_as_float(bin_img)

    return float_img, orig, ihc  # , float_masked


def punch(img):
    # Identifiying the Tissue punches in order to Crop the image correctly
    # Canny edges and RANSAC is used to fit a circe to the punch
    # A Mask is created

    distance = 0
    r = 0

    float_im, orig, ihc = create_bin(img)
    gray = rgb2grey(orig)                   # Original Image in grey scale
    smooth = gaussian(gray, sigma=3)

    shape = np.shape(gray)                 # Gives size of image
    l = shape[0]
    w = shape[1]

    x = l - 40
    y = w - 40

    rows = np.array([np.arange(x, x+20)])       # accesing bottom right corner of image to create threshold from
    columns = np.array([np.arange(y, y+20)])

    corner = gray[rows, columns]

    thresh = np.mean(corner)
    print thresh
    binar = (smooth < thresh - 0.01)

    bin = remove_small_holes(binar, min_size=100000, connectivity=2)    # Removes backround form inside punch
    bin1 = remove_small_objects(bin, min_size=5000, connectivity=2)     # removes small tissue objects
    bin2 = gaussian(bin1, sigma=3)      # smooth image
    bin3 = (bin2 > 0)       # changing from float to boolean

    # eosin = IHC[:, :, 2]

    edges = canny(bin3)     # edges of binary image
    coords = np.column_stack(np.nonzero(edges))     # coordinate of the edges above

    model, inliers = ransac(coords, CircleModel, min_samples=4, residual_threshold=1, max_trials=1000)
    #  circle fitting around core the more samples the "better" the approximation... may be too tight

    a, b = model.params[0], model.params[1]     # centroid of approximated circle
    r = model.params[2]     # radius of approximated circle
    ny, nx = bin3.shape     # nx, ny is size of entire image
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))      # return coordinate matrices from coordinate vectors
    distance = np.sqrt((ix - b)**2 + (iy - a)**2)       # distance is a vector array of distances to each point in img

    return distance, r, float_im, orig, ihc, bin3


def label_img(img):
    # Labelling the nests is done using connected components
    dist = 0
    radius = 0
    dist, radius, float_img, orig, ihc, bin3 = punch(img)
    masked_img = np.ma.masked_where(dist > radius, float_img)   # mask image when the distance is greater than radius
    masked_bool = np.ma.filled(masked_img, fill_value=0)    # fill mask with 0 values

    min_nest_size, min_hole_size = parameters.params()

    labeled_img = label(input=masked_bool, connectivity=2, background=0)    # label function to connect components
    rem_holes = remove_small_holes(labeled_img, min_size=min_hole_size, connectivity=2)     # removes holes in nest
    labeled_img1 = remove_small_objects(rem_holes, min_size=min_nest_size, connectivity=2)  # remove small nests
    labeled = label(labeled_img1, connectivity=2, background=0)     # re-label nests without holes or small nests

    print labeled

    return labeled, masked_img, orig, ihc, bin3, float_img


def display_image(img):
    # Displaying images if needed

    labeled_img, masked_img, orig, ihc, bin3, float_img = label_img(img)
    n = len(np.unique(labeled_img)) - 1

    plt.figure()
    plt.subplot(141)
    plt.imshow(ihc[:, :, 2], cmap='gray')
    plt.title("DAB color space")

    plt.subplot(142)
    plt.imshow(labeled_img, cmap='spectral')
    plt.title("Labeled Image %d" % n)

    plt.subplot(143)
    plt.imshow(mark_boundaries(orig, label_img=labeled_img, color=(1, 0, 0)))
    plt.title('Overlay Outlines')

    plt.subplot(144)
    plt.imshow(masked_img, cmap='gray')
    plt.title('Full Punch Segmentation')


def get_data(img):
    # Obtaining the data for each nest
    cf = 0.53
    labels, mask, orig, ihc, bin3, float_img = label_img(img)
    props = regionprops(labels)
    pixel_res, pixel_size = parameters.microns()

    area = []
    perimeter = []
    eccentricity = []
    filled_area = []
    maj_axis = []
    min_axis = []

    ns = len(np.unique(labels)) - 1     # number of nests -1 as 0 is not a nest
    print ns, 'Number of Nests'

    for seg in range(ns):       # seg is each nest in core/punch

        area.append(props[seg].area)
        perimeter.append(props[seg].perimeter)
        eccentricity.append(props[seg].eccentricity)
        filled_area.append(props[seg].filled_area)
        min_axis.append(props[seg].minor_axis_length)
        maj_axis.append(props[seg].major_axis_length)

    avg_area = np.mean(area)
    avg_perimeter = np.mean(perimeter)
    avg_eccentricity = np.mean(eccentricity)
    avg_filled_area = np.mean(filled_area)
    roundness = map(truediv, min_axis, maj_axis)
    new_list = [4 * pi * a for a in area]
    circularity = map(truediv, new_list, (np.square(perimeter)))
    avg_roundness = np.mean(roundness)
    avg_circularity = np.mean(circularity)

    total_nest_area = sum(area)
    total_nest_perim = sum(perimeter)
    std_dev_area = np.std(area)
    std_dev_perimeter = np.std(perimeter)
    std_dev_eccentricity = np.std(eccentricity)
    std_dev_filled_area = np.std(filled_area)
    std_dev_roundness = np.std(roundness)
    std_dev_circularity = np.std(circularity)
    name = os.path.basename(os.path.normpath(img))

    area_microns = [pixel_size * a for a in area]
    avg_area_microns = np.mean(area_microns)
    perimeter_microns = [pixel_res * p for p in perimeter]
    avg_perimeter_microns = np.mean(perimeter_microns)
    new_list1 = [4 * pi * a for a in area_microns]
    circularity_micron = map(truediv, new_list1, (np.square(perimeter_microns)))

    return ns, area, perimeter, eccentricity, filled_area, avg_area, avg_perimeter, avg_eccentricity, avg_filled_area,\
        roundness, circularity, avg_roundness, avg_circularity, total_nest_area, total_nest_perim,\
        std_dev_area, std_dev_perimeter, std_dev_eccentricity, std_dev_filled_area, std_dev_roundness,\
        std_dev_circularity, name, area_microns, avg_area_microns, perimeter_microns, avg_perimeter_microns,\
        circularity_micron


def write_csv(output_data, save_path, micron=False):
    # Writing the data file to as CSV
    if micron:
        name = 'output_micron_data'
    else:
        name = 'output_data'

    save_out = save_path + '/%s.csv' % name

    with open(save_out, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output_data)


def save_image(save_path, img):
    # If needed the images can be saved for further analysis of examination
    original_img, ihc_img = color_conversion(img)
    l_img, masked_img, orig, ihc, bin3, float_img = label_img(img)
    orig = os.path.basename(os.path.normpath(img))
    # plt.imshow(mark_boundaries(orig, label_img=labeled_img, color=(1, 0, 0)))
    img_file = mark_boundaries(original_img, label_img=l_img, color=(1, 0, 0))
    # img_file = l_img
    filename = save_path + '/Segments' + orig  # 'Labelled_%s' % + ".png"  # '%s' % img
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(filename)
    plt.imsave(fname=filename, arr=img_file)

    img_file1 = l_img
    filename = save_path + '/labelled' + orig
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(filename)
    plt.imsave(fname=filename, arr=img_file1, cmap='spectral')
