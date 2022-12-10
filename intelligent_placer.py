import os
import numpy as np
import cv2
from itertools import permutations
from copy import deepcopy
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import binary_closing


def compress_image(image):  # сжатие
    new_height = int(image.shape[0] * 50 / 100)
    new_width = int(image.shape[1] * 50 / 100)
    compressed_image = cv2.resize(image, (new_width, new_height), cv2.INTER_AREA)
    return compressed_image


def prepare_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # изменение палитры
    binary_image = 255 * binary_fill_holes(binary_closing(canny(gray_image, sigma=3), footprint=np.ones((20, 20))))
    binary_image = binary_image.astype(np.uint8)
    return binary_image


def height_coordinates(points):  # поиск крайних северной и южной точек
    north = south = points[0]

    for extreme_point in points[1:]:
        if extreme_point[1] < north[1]:
            north = extreme_point
        elif extreme_point[1] > south[1]:
            south = extreme_point
    return north, south


def width_coordinates(points):  # поиск крайних западной и восточной точек
    west = east = points[0]

    for extreme_point in points[1:]:
        if extreme_point[0] < west[0]:
            west = extreme_point
        elif extreme_point[0] > east[0]:
            east = extreme_point
    return west, east


def find_contours(image):  # ищем контуры и сохраняем как массив точек
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    ready_contours = []
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * epsilon, True)
        ready_contours.append([point[0] for point in approx])
    return ready_contours


def separate_polygon(contours): # отделяем предметы от многоугольника
    polygon_sep = 0
    min_x = contours[0][0][0]
    for i in range(len(contours)):
        for point in contours[i]:
            if point[0] < min_x:
                min_x = point[0]
                polygon_sep = i

    polygon = contours[polygon_sep]
    objects = contours[:polygon_sep] + contours[polygon_sep + 1:]
    return polygon, objects


def draw_contours(image, contour, color, fat):
    curr_object = []
    for i in range(len(contour)):
        curr_object.append([[contour[i][0], contour[i][1]]])
    cv2.drawContours(image, [np.array(curr_object)], -1, color, fat)


def point_inside(point_x, point_y, curr_object):  # определение принадлежности точки объекту
    check_inside = 0
    object_x = [point[0] for point in curr_object]
    object_y = [point[1] for point in curr_object]
    # из заданной точки выходит горизонтальный луч, считаем число пересечений луча со сторонами
    # если число пересечений нечетное, получаем принадлежность
    for i in range(len(object_y)):
        if ((object_y[i] <= point_y and point_y < object_y[i - 1] or object_y[i - 1] <= point_y and point_y < object_y[i])
                and (point_x > (object_x[i - 1] - object_x[i]) * (point_y - object_y[i]) / (object_y[i - 1] - object_y[i]) +
                     object_x[i])):
            check_inside = 1 - check_inside
    if check_inside:
        return True
    return False


def find_place(polygon, curr_object, placed_objects):
    check_place = True

    north_polygon, south_polygon = height_coordinates(polygon)
    north_object, south_object = height_coordinates(curr_object)
    west_polygon, east_polygon = width_coordinates(polygon)
    west_object, east_object = width_coordinates(curr_object)

    height_polygon = south_polygon[1] - north_polygon[1]
    height_object = south_object[1] - north_object[1]
    width_polygon = east_polygon[0] - west_polygon[0]
    width_object = east_object[0] - west_object[0]

    move_north = north_object[1] - north_polygon[1]
    move_east = east_object[0] - east_polygon[0]

    if height_polygon < height_object or width_polygon < width_object:
        check_place = False
        return check_place, []
    # сдвиг на северо-запад
    for i in range(len(curr_object)):
        curr_object[i][1] -= move_north
        curr_object[i][0] -= move_east

    while west_polygon[0] > west_object[0]:
        while south_polygon[1] > south_object[1]:
            check_place = True
            for point in curr_object:
                if not point_inside(point[0], point[1], polygon):  # предмет за пределами многоугольника
                    check_place = False
                    break
                for placed_object in placed_objects:
                    if point_inside(point[0], point[1],
                                    placed_object):  # предмет накладывается на другие предметы внутри многоугольника
                        check_place = False
                        break
            if check_place:
                return check_place, curr_object
            for i in range(len(curr_object)):
                curr_object[i][1] += 1

        north_object, south_object = height_coordinates(curr_object)
        move_north = north_object[1] - north_polygon[1]
        for i in range(len(curr_object)):
            curr_object[i][1] -= move_north  # сдвиг обратно вверх при невыполнении условий
            curr_object[i][0] += 1

    check_place = False

    return check_place, []


def make_full_placement(polygon, objects):
    all_options = list(permutations(objects))
    placed_objects = []

    for curr_option in all_options:
        check_places = True
        for object in curr_option: # находим место каждого предмета добавляем его к уже размещенным
            found_place, object_position = find_place(polygon, deepcopy(object), placed_objects)
            placed_objects.append(object_position)
            if not found_place:
                check_places = False
                break
        if check_places:
            return True, placed_objects

    check_places = False

    return check_places, []


def placer(image, polygon, objects):
    # планируется основной метод для размещения предметов в прямоугольнике. Сейчас здесь каркас
    good_placed = "Yes"
    bad_placed = "No"

    placed = make_full_placement(image, polygon, curr_object)
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


if __name__ == '__main__':
    image = cv2.imread('cases\\case1.jpg')
    compressed_image = compress_image(image)
    binary_image = prepare_image(compressed_image)

    objects_contours = find_contours(binary_image)
    red_color = (255, 0, 0)

    for curr_object in objects_contours:
        draw_contours(compressed_image, curr_object, red_color, 6)

    polygon, objects = separate_polygon(objects_contours)

    draw_contours(compressed_image, polygon, red_color, 6)
    blue_color = (0, 0, 255)
    for curr_object in objects:
        draw_contours(compressed_image, curr_object, blue_color, 6)

    found_placement, placed_objects = make_full_placement(polygon, objects)
    print(found_placement)