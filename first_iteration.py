import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import binary_closing
import os


def prepare_image(image):
    new_height = int(image.shape[0] * 30 / 100)
    new_width = int(image.shape[1] * 30 / 100)
    compressed_image = cv2.resize(image, (new_width, new_height), cv2.INTER_AREA)

    gray_image = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2GRAY)
    binary_image = 255 * binary_fill_holes(binary_closing(canny(gray_image, sigma=3), selem=np.ones((20, 20))))
    binary_image = binary_image.astype(np.uint8)
    return binary_image


def find_contours(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    ready_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
        ready_contours.append([point[0] for point in approx])
    return ready_contours


def height_coordinates(points):
    north = south = points[0]

    for extreme_point in points[1:]:
        if extreme_point[1] < north[1]:
            north = extreme_point
        elif extreme_point[1] > south[1]:
            south = extreme_point
    return north, south


def width_coordinates(points):
    west = east = points[0]

    for extreme_point in points[1:]:
        if extreme_point[0] < west[0]:
            west = extreme_point
        elif extreme_point[0] > east[0]:
            east = extreme_point
    return west, east


def find_place(image, polygon, curr_object):
    # планируется метод для поиска места конкретного предмета
    north_polygon, south_polygon = height_coordinates(polygon)
    north_object, south_object = height_coordinates(curr_object)
    west_polygon, east_polygon = width_coordinates(polygon)
    west_object, east_object = width_coordinates(curr_object)

    height_polygon = south_polygon[1] - north_polygon[1]
    height_object = south_object[1] - north_object[1]
    width_polygon = east_polygon[0] - west_polygon[0]
    width_object = east_object[0] - west_object[0]

    if height_polygon < height_object or width_polygon < width_object:
        return 0

    # здесь планируется перенос предмета на северо-запад многоугольника
    #
    #
    # здесь планируется сам алгоритм поиска подходящего места через попиксельный сдвиг вниз и вправо
    #
    #
    #

def placer(image, polygon, objects):
    # планируется основной метод для размещения предметов в прямоугольнике. Сейчас здесь каркас
    good_placed = "Yes"
    bad_placed = "No"

    for curr_object in objects:
        placed = find_place(image, polygon, curr_object)
    if placed == 0:
        return bad_placed
    else:
        return good_placed


def errors_processing(data, number_of_error):
    if number_of_error == 1:
        if not os.path.exists(data[0]):
            print('Incorrect path')
            return 0
        image = cv2.imread(data[0])
        if image is None:
            print('Can\'t read the image')
            return 0

    if number_of_error == 2:
        if len(data[0]) > 5:
            print('Too much vertices of polygon')
            return 0
        if not data[1]:
            print('No objects found')
            return 0
