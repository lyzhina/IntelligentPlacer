import os
import numpy as np
import cv2
from itertools import permutations
from copy import deepcopy
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import binary_closing
from matplotlib import pyplot as plt


def compress_image(image):  # сжатие изображения
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
            if point[0] < min_x: # ищем многоугольник как объект с наименьшей x-координатой
                min_x = point[0]
                polygon_sep = i

    polygon = contours[polygon_sep]
    objects = contours[:polygon_sep] + contours[polygon_sep + 1:]
    return polygon, objects


def draw_contours(image, contour, color, thickness): # отрисовка контуров
    curr_object = []
    for i in range(len(contour)):
        curr_object.append([[contour[i][0], contour[i][1]]])
    cv2.drawContours(image, [np.array(curr_object)], -1, color, thickness)


def point_inside(point_x, point_y, curr_object):  # определение принадлежности точки объекту
    check_inside = False
    object_x = [point[0] for point in curr_object]
    object_y = [point[1] for point in curr_object]
    # из заданной точки выходит горизонтальный луч, считаем число пересечений луча со сторонами
    # если число пересечений нечетное, получаем принадлежность
    for i in range(len(object_y)):
        # первая строка условия: попадание y-координаты заданной точки между y-координатами точек многоугольника +
        # + направление движения + ненулевой знаменатель в строке ниже
        # вторая строка условия: сторона многоугольника слева от точки
        if ((object_y[i] <= point_y and point_y < object_y[i - 1] or object_y[i - 1] <= point_y and point_y < object_y[i])
                and (point_x > (object_x[i - 1] - object_x[i]) * (point_y - object_y[i]) / (object_y[i - 1] - object_y[i]) +
                     object_x[i])):
            check_inside = not check_inside
    if check_inside:
        return True
    return False


def find_place(polygon, curr_object, placed_objects):
    check_place = True
    # поиск крайних точек
    north_polygon, south_polygon = height_coordinates(polygon)
    north_object, south_object = height_coordinates(curr_object)
    west_polygon, east_polygon = width_coordinates(polygon)
    west_object, east_object = width_coordinates(curr_object)
    # поиск длины / ширины
    height_polygon = south_polygon[1] - north_polygon[1]
    height_object = south_object[1] - north_object[1]
    width_polygon = east_polygon[0] - west_polygon[0]
    width_object = east_object[0] - west_object[0]
    # величина сдвига на северо-восток
    move_north = north_object[1] - north_polygon[1]
    move_west = west_object[0] - west_polygon[0]

    if height_polygon < height_object or width_polygon < width_object:
        check_place = False
        return check_place, []
    # сдвиг на северо-запад
    for i in range(len(curr_object)):
        curr_object[i][1] -= move_north
        curr_object[i][0] -= move_west

    while east_object[0] < east_polygon[0]:  # ищем место, пока не дошли до востока многоугольника
        while south_object[1] < south_polygon[1]: # сдвиг по пикселю вниз, пока не дошли до юга многугольника
            check_place = True
            for point in curr_object: # для каждой точки текущего предмета проверяем, что она не вышла за пределы и не наложилась на другие предметы
                if not point_inside(point[0], point[1], polygon): # предмет за пределами многоугольника
                    check_place = False
                    break
                for placed_object in placed_objects: # точка текущего предмета оказалась внутри другого предмета
                    if point_inside(point[0], point[1], placed_object):
                        check_place = False
                        break
                    for placed_point in placed_object: # точка другого предмета оказалась внутри текущего
                        if point_inside(placed_point[0], placed_point[1], curr_object):
                            check_place = False
                            break

            if check_place:
                return check_place, curr_object

            for i in range(len(curr_object)): # если предмет не встал, сдвигаем его на пиксель вниз
                curr_object[i][1] += 1

        move_north = north_object[1] - north_polygon[1]
        for i in range(len(curr_object)): # сдвиг обратно вверх при невыполнении условий
            curr_object[i][0] += 1
            curr_object[i][1] -= move_north

    check_place = False  # место не нашлось
    return check_place, []


def make_full_placement(polygon, objects):
    all_options = list(permutations(objects))
    placed_objects = []
    for curr_option in all_options:
        check_places = True
        for object in curr_option: # находим место каждого предмета, добавляем его к уже размещенным
            found_place, object_position = find_place(polygon, deepcopy(object), placed_objects)
            placed_objects.append(object_position)
            if not found_place:
                check_places = False
                break
        if check_places:
            return True, placed_objects

    check_places = False

    return check_places, []


def intelligent_placer(path_image):
    good_placed = "Yes"
    bad_placed = "No"
    if not os.path.exists(path_image):
        print('Incorrect path')
        return
    image = cv2.imread(path_image)

    if image is None:
        print('Can\'t read the image')
        return
    compressed_image = compress_image(image)
    binary_image = prepare_image(compressed_image)
    objects_contours = find_contours(binary_image)
    polygon, objects = separate_polygon(objects_contours)

    red_color = (255, 0, 0)
    blue_color = (0, 0, 255)
    draw_contours(compressed_image, polygon, red_color, 5)
    for curr_object in objects:
        draw_contours(compressed_image, curr_object, blue_color, 5)

    found_placement, placed_objects = make_full_placement(polygon, objects)
    if found_placement:
        print(good_placed)
        green_color = (0, 255, 0)
        for placed_object in placed_objects:
            draw_contours(compressed_image, placed_object, green_color, 3)
    else:
        print(bad_placed)

    plt.imshow(compressed_image)

if __name__ == '__main__':
    intelligent_placer('cases\\case5.jpg')
