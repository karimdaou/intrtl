# intrtl
Random rotation and brightness adjustment using OpenCV

Необходимо написать скрипт на языке python для аугментации изображений. Функции (или методы класса) должны быть реализованы так, чтобы их можно было удобно внедрить в реальные проекты. То есть написаны понятно и структурировано, и, конечно же, без фанатизма. Нельзя использовать сторонние библиотеки/модули для аугментации. Можно использовать opencv.

	Виды аугментаций:
1. Поворот изображения
2. Повышение и понижение яркости

	Дано:
1. Изображения
2. Координаты объектов на изображениях в формате pickle
2.1. Координаты хранятся в следующей последовательности x1, y1, x2, y2

	Реализовать обязательно:
1. У скрипта должны быть следующие изменяемые параметры:
1.1. Кол-во аугментированных фото на выходе для каждого файла.
1.2. Шанс срабатывания каждого вида аугментации.
1.3. Пороги для максимального поворота в одну и другую сторону в градусах.
1.4. Пороги на максимального повышения и понижения яркости.
1.5. Возможность визуализации разметки при сохранении в виде отдельного флага.
2. Возможность обработать разный по размеру список изображений.
3. При повороте изображения координаты должны смещаться соответственно объектам.
4. Аугментированные изображения должны сохраняться в отдельную папку.
5. Разметка к аугментированным изображениям должна сохраняться  всё в одном файле или в отдельных pickle-файлах (или json-файлах) для каждого изображения. Разметка должна сохраняться в таком же формате как в исходном pickle-файле.
6. Реализация повышения и понижения яркости без использования циклов с проходом по всем пикселям. То есть с использованием numpy-функций и/или cv2-функций (opencv).
