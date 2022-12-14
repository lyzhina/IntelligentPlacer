# IntelligentPlacer
## Постановка задачи
На вход подается фотография, на которой изображены различные ранее выбранные предметы на светлой горизонтальной поверхности и белый лист бумаги формата A4. На листе нарисован многоугольник.  
Нужно определить, можно ли поместить предметы внутрь многоугольника так, чтобы они не перекрывали друг друга.
## Требования к входным данным
### Фотометрические требования:
1) Допустимый формат для фотографий - .jpg;
2) Съемка предметов производится сверху, перпендикулярно плоскости предметов (возможно незначительное отклонение - не более 10 градусов), на расстоянии 30-40 см от предметов;
3) Изображения цветные.
### Требования по расположению объектов:
1) Объекты полностью попадают в кадр;
2) Предметы в кадре не перекрывают друг друга;
3) Предметы по площади не перекрывают площадь листа формата A4, не перекрывают границы листа;
4) Каждый предмет имеет четкие границы, которые не сливаются с фоном. Расстояние между границами предметов не менее 0.5 см;
5) В случае полых предметов (см. предмет 6 - резинка) в полой части не должны находиться другие предметы;
6) Цвет каждого предмета контрастен с цветом фона и с цветами других предметов;
7) Многоугольник на листе выпуклый и имеет не более 5 вершин;
8) Многоугольник расположен слева от предметов;
9) Многоугольник на листе нарисован отчетливо, темным маркером. Толщина линии не менее 1 мм.
### Требования к поверхности:
1) Поверхность одна и та же на всех фотографиях;
2) Поверхность ровная.
## Требования к выходным данным:
В результате работы алгоритма для каждого кейса в консоль выводится ответ:
- "Yes" - если предметы помещаются;
-  "No" - если уместить предметы нельзя.
## Данные
Фотографии предметов и поверхности располагаются в папке data.  
Фотографии кейсов располагаются в папке cases. В этой же папке располагается файл expected_result.md с описанием ожидаемого вывода результата для каждого кейса.
## План
- Подготовка:
  + Сжатие изображения для ускорения операций;
  + Изменение палитры изображения на grayscale с помощью cv2.cvtColor ;
- Идентификация:
  + Бинаризация при помощи детектора Canny и морфологических операций (замыкание) с использованием OpenCv;
  + Выделение контуров с помощью cv2.findContours;
  + *возможно, стоит сохранить границы как массив точек и дальше работать с ним*
  + Отделение многоугольника от предметов. У многоугольника должна быть наименьшая x-координата, т.к. он расположен слева;
- Распределение:
  + Сортировка предметов по убыванию площади;
  + Для каждого предмета от больших к меньшим:
  + Цикл (принцип: от северо-запада к юго-востоку):
    + Нахождение длины и ширины предмета и многоугольника как разницу крайних точек (север / юг / запад / восток). Определение того, что длина/ширина многоугольника превышает соответствующие параметры предмета. Иначе на выход сразу "No";
    + Параллельный перенос предмета влево и вверх до встречи с крайними точками многоугольника. Сдвигом на расстояние "запад предмета - запад многоугольника" и "север предмета - север многоугольника" получается достигнуть такого соприкосновения;
    + Перенос предмета на пиксель вниз до встречи в южной точкой при условии, что (1) предмет не накладывается на другие предметы внутри многоугольника и (2) предмет полностью находится внутри многоугольника; (уточнение (*))
    +  Если (1) или (2) не обеспечены, возвращение обратно вверх и перенос по тому же принципу вправо;
    +  Если предмет не нашел свое место, пройдя через многоугольник, на выход "No";
  + При положительном исходе общего размещения на выход "Yes".
  + (*) *для полых предметов (например, резинка) в начальной реализации алгоритма считаем основной границей предмета внешнюю, то есть внутри полости помещать другие предметы нельзя*
  + (* *) *после успешной реализации намеченного плана планирую ввести концепцию поворота предметов для оптимизации пространства. Тогда вычисление + сравнение длины/ширины предмета и многоугольника будут происходить на каждом шаге поворота*
