import cv2
import imutils
import numpy as np
from pyzbar import pyzbar

# Cargar la imagen
# img = cv2.imread('img/test/barcode_detector_test_025565.jpg', cv2.IMREAD_COLOR) #1º
# img = cv2.imread('img/test/Barcodes_val_010985.jpg', cv2.IMREAD_COLOR) #-45º
# img = cv2.imread('img/test/barcode_detector_train_025407.jpg', cv2.IMREAD_COLOR) #87º
# img = cv2.imread('img/test/thisisnotgood_val_016289.jpg', cv2.IMREAD_COLOR) #-23º
img = cv2.imread('img/train/barcode_detector_test_025552.jpg', cv2.IMREAD_COLOR) #133º
# img = cv2.imread('images/barcode_01.jpg', cv2.IMREAD_COLOR) #47º
# img = cv2.imread('images/barcode_02.jpg', cv2.IMREAD_COLOR) #-23º
# img = cv2.imread('images/barcode_03.jpg', cv2.IMREAD_COLOR) #-45º
# img = cv2.imread('images/barcode_04.jpg', cv2.IMREAD_COLOR) #87º
# img = cv2.imread('images/barcode_05.jpg', cv2.IMREAD_COLOR) #-133º
# img = cv2.imread('images/barcode_06.jpg', cv2.IMREAD_COLOR) #1º
print(img.shape[1])
# Definir los factores de escala
if 1600 > img.shape[1] > 1000:
    scale_x = 0.3
    scale_y = 0.3
elif img.shape[1] >= 1600:
    scale_x = 0.25
    scale_y = 0.25
else:
    scale_x = 1
    scale_y = 1

# Redimensionar la imagen utilizando factores de escala
resized_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

# Convertir a escala de grises
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Bordes Canny', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow('Gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Aplicar el detector de bordes Canny
# bordes = cv2.Canny(gray, 10, 240, L2gradient = True)

# # Mostrar los bordes detectados
# cv2.imshow('Bordes Canny', bordes)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Transformada de Hough para detectar líneas
# lines = cv2.HoughLinesP(bordes, 1, np.pi/180, 40, minLineLength=30, maxLineGap=15)

# # Vamos a obter la orientación dominante de las líneas detectadas
# if lines is not None:
#     all_angles = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         dx = x2-x1
#         dy = y2-y1
#         angle = np.degrees(np.arctan2(dy, dx))
#         all_angles.append(angle)

#     dominant_angle = np.median(all_angles)

#     if -90 <= dominant_angle < 0:
#         rotacion = dominant_angle #-47º
#     elif 0 <= dominant_angle < 90:
#         rotacion = 90 - dominant_angle #47º
#     elif -180 < dominant_angle < -90:
#         rotacion = (180 + dominant_angle) + 90 #-133º
#     else:
#         rotacion = 90 - (180 - dominant_angle) #133º

# # Calculamos el centro de la imagen para la rotación
# alto, ancho = resized_image.shape[0:2]
# centro = (ancho // 2, alto // 2)
# angulo = rotacion
# escala = 1.0

# # Matriz de rotacion
# matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, escala)

# # Aplicamos la rotacion
# imagen_rotada = cv2.warpAffine(resized_image, matriz_rotacion, (ancho, alto))

# # Mostrar la imagen con las líneas detectadas
# cv2.imshow('Deteccion de lineas', imagen_rotada)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Usamos Canny directamente sobre la imagen rotada
# gray_rot = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imshow("Imagen tras umbralizar", thresh)
cv2.waitKey(0)

# Closing morfológico para unir las barras en una masa
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Limpieza básica
closed = cv2.dilate(closed, None, iterations=4)
closed = cv2.erode(closed, None, iterations=4)

cv2.imshow("Closed (solo con Canny)", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Encontrar contornos en la imagen procesada
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# Hacer un bounding box alrededor del contorno más grande
rect = cv2.minAreaRect(c)
print(rect)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = box.astype(int)
print(box)

'''Codigo interesatne para mostrar como funciona boxpoints
for i in box:
    cv2.circle(image,(i[0],i[1]), 3, (0,255,0), -1)
    imgplot = plt.imshow(image)
    plt.show()
'''

#Vamos a guardar la rotacion, altura y anchura para poner la imagen derecha
angulo = rect[2]
alto, ancho = resized_image.shape[0:2]
if angulo < -45:
    angulo = -(90 + angulo)
else:
    angulo = angulo - 90
matriz_rotacion = cv2.getRotationMatrix2D(rect[0], angulo, 1.0)
imagen_rotada = cv2.warpAffine(resized_image, matriz_rotacion, (ancho, alto))

pts_rotados = cv2.transform(np.array([box]), matriz_rotacion)[0]

# debug = imagen_rotada.copy()
cv2.drawContours(imagen_rotada, [pts_rotados], -1, (0, 255, 0), 3)
cv2.imshow("Contorno seleccionado", imagen_rotada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bounding box tras rotar
xs = pts_rotados[:, 0]
ys = pts_rotados[:, 1]

# Cogemos las coordenadas minimas y maximas ya que sino la imagen recortada sale mal
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

# Recortamos la imagen
imagen_recortada = imagen_rotada[y_min:y_max, x_min:x_max]

cv2.imshow("Imagen recortada", imagen_recortada)
cv2.waitKey(0)
cv2.destroyAllWindows()

