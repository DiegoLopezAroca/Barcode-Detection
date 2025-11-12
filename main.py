import cv2
import numpy as np

# Cargar la imagen
#img = cv2.imread('img/test/barcode_detector_test_025565.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('img/test/Barcodes_val_010985.jpg', cv2.IMREAD_COLOR)

# Definir los factores de escala
scale_x = 0.5
scale_y = 0.5

# Redimensionar la imagen utilizando factores de escala
resized_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

# Convertir a escala de grises
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Aplicar el detector de bordes Canny
bordes = cv2.Canny(gray, 50, 220, L2gradient = True)

# Mostrar los bordes detectados
cv2.imshow('Bordes Canny', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Transformada de Hough para detectar líneas
'''
100: Umbral. Solo se devolverán las líneas que tengan al menos este valor de votos.
minLineLength: Longitud mínima de una línea. Las líneas más cortas que esto se rechazan.
maxLineGap: Máxima distancia entre segmentos de línea que se considerarán como una sola línea.
'''
lines = cv2.HoughLinesP(bordes, 1, np.pi/180, 40, minLineLength=20, maxLineGap=15)

bordes = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)

# Vamos a obter la orientación dominante de las líneas detectadas
if lines is not None:
    all_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(bordes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Pendiente
        dx = x2-x1
        dy = y2-y1
        angle = np.degrees(np.arctan2(dy, dx))
        all_angles.append(angle)

    angles = [(a + 180) % 180 for a in all_angles]
    dominant_angle = np.median(angles)

    if 0 < dominant_angle < 90 or -180 < dominant_angle < -90:
        rotacion = dominant_angle - (dominant_angle - 90)
    else:
        rotacion = dominant_angle - 90

# Calculamos el centro de la imagen para la rotación
alto, ancho = resized_image.shape[0:2]
centro = (ancho // 2, alto // 2)
angulo = rotacion
escala = 1.0

# Matriz de rotacion
matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, escala)

# Aplicamos la rotacion
imagen_rotada = cv2.warpAffine(resized_image, matriz_rotacion, (ancho, alto))

# Reescalamos la imagen

# Mostrar la imagen con las líneas detectadas
cv2.imshow('Deteccion de lineas', imagen_rotada)
cv2.waitKey(0)
cv2.destroyAllWindows()