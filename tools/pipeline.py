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
    # original_img, mask, dab, rr, cc = segment(img)
    # b = mask.data
    # rescaled_mask = rescale_intensity(b, out_range=(0, 1))
    orig, ihc = color_conversion(img)
    rescaled = rescale_intensity(ihc[:, :, 2], out_range=(0, 1))
    int_img = img_as_uint(rescaled)

    # int_mask_data = img_as_uint(rescaled_mask)
    # print 'loop once'
    return int_img, orig, ihc  # , int_mask_data


def create_bin(img):  # otsu_method=True):
    # Binary image created from Threshold, then labelling is done on this image
    # if otsu_method:
    int_img, orig, ihc = rescale(img)
    t_otsu = threshold_otsu(int_img)
    bin_img = (int_img >= t_otsu)
    # bin =

    float_img = img_as_float(bin_img)

    # float_masked = img_as_float(bin_masked)

    return float_img, orig, ihc  # , float_masked


def punch(img):
    # Identifiying the Tissue punches in order to Crop the image correctly
    # Canny edges and RANSAC is used to fit a circe to the punch
    # A Mask is created

    distance = 0
    r = 0

    float_im, orig, ihc = create_bin(img)
    gray = rgb2grey(orig)
    smooth = gaussian(gray, sigma=3)

    shape = np.shape(gray)
    l = shape[0]
    w = shape[1]

    x = l - 20
    y = w - 20

    rows = np.array([[x, x, x, x], [x + 1, x + 1, x + 1, x + 1]])
    columns = np.array([[y, y, y, y], [y + 1, y + 1, y + 1, y + 1]])

    corner = gray[rows, columns]

    thresh = np.mean(corner)
    print thresh
    binar = (smooth < thresh - 0.01)

    bin = remove_small_holes(binar, min_size=100000, connectivity=2)
    bin1 = remove_small_objects(bin, min_size=5000, connectivity=2)
    bin2 = gaussian(bin1, sigma=3)
    bin3 = (bin2 > 0)

    # eosin = IHC[:, :, 2]
    edges = canny(bin3)
    coords = np.column_stack(np.nonzero(edges))

    model, inliers = ransac(coords, CircleModel, min_samples=4, residual_threshold=1, max_trials=1000)

    # rr, cc = circle_perimeter(int(model.params[0]),
    #                          int(model.params[1]),
    #                          int(model.params[2]),
    #                          shape=im.shape)

    a, b = model.params[0], model.params[1]
    r = model.params[2]
    ny, nx = bin3.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    distance = np.sqrt((ix - b)**2 + (iy - a)**2)

    mask = np.ma.masked_where(distance > r, bin3)

    return distance, r, float_im, orig, ihc, bin3


def label_img(img):
    # Labelling the nests is done using connected components
    dist = 0
    radius = 0
    dist, radius, float_img, orig, ihc, bin3 = punch(img)
    masked_img = np.ma.masked_where(dist > radius, float_img)
    masked_bool = np.ma.filled(masked_img, fill_value=0)

    min_nest_size = 100  # Size in Pixels of minimum nest
    min_hole_size = 500  # Size in Pixels of minimum hole
    min_nest_size, min_hole_size = parameters.params()


    labeled_img = label(input=masked_bool, connectivity=2, background=0)
    rem_holes = remove_small_holes(labeled_img, min_size=min_hole_size, connectivity=2)
    labeled_img1 = remove_small_objects(rem_holes, min_size=min_nest_size, connectivity=2)
    labeled = label(labeled_img1, connectivity=2, background=0)
    mask_lab = np.ma.masked_where(dist > radius, labeled)

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


    #return plt.show()


def get_data(img):
    # Obtaining the data for each nest
    cf = 0.53
    labels, mask, orig, ihc, bin3, float_img = label_img(img)
    props = regionprops(labels)

    area = []
    perimeter = []
    eccentricity = []
    filled_area = []
    maj_axis = []
    min_axis = []

    ns = len(np.unique(labels)) - 1
    print ns, 'Number of Nests'

    for seg in range(ns):

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

    # micron_area = [x * 0.1 for x in area]

    return ns, area, perimeter, eccentricity, filled_area, avg_area, avg_perimeter, avg_eccentricity, avg_filled_area,\
        roundness, circularity, avg_roundness, avg_circularity, total_nest_area, total_nest_perim,\
        std_dev_area, std_dev_perimeter, std_dev_eccentricity, std_dev_filled_area, std_dev_roundness,\
        std_dev_circularity, name


def write_csv(output_data, save_path):
    # Writing the data file to as CSV
    save_out = save_path + '/output_data.csv'

    with open(save_out, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output_data)


def save_image(save_path, img):  # overlay=True, binary=False, DAB=False):
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
