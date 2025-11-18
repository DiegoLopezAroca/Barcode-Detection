import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('img/test/barcode_detector_test_025565.jpg', cv2.IMREAD_COLOR) #1º
img = cv2.imread('img/test/Barcodes_val_010985.jpg', cv2.IMREAD_COLOR) #-45º
img = cv2.imread('img/test/barcode_detector_train_025407.jpg', cv2.IMREAD_COLOR) #87º
#img = cv2.imread('img/test/thisisnotgood_val_016289.jpg', cv2.IMREAD_COLOR) #-23º

# Definir los factores de escala
scale_x = 0.3
scale_y = 0.3

# Redimensionar la imagen utilizando factores de escala
resized_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

# Convertir a escala de grises
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Aplicar el detector de bordes Canny
bordes = cv2.Canny(gray, 10, 240, L2gradient = True)

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
lines = cv2.HoughLinesP(bordes, 1, np.pi/180, 40, minLineLength=30, maxLineGap=15)

bordes_color = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)

# Vamos a obter la orientación dominante de las líneas detectadas
if lines is not None:
    all_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #cv2.line(bordes_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Pendiente
        dx = x2-x1
        dy = y2-y1
        angle = np.degrees(np.arctan2(dy, dx))
        all_angles.append(angle)

    dominant_angle = np.median(all_angles)

    if -90 <= dominant_angle < 0:
        rotacion = dominant_angle #-47º
    elif 0 <= dominant_angle < 90:
        rotacion = 90 - dominant_angle #47º
    elif -180 < dominant_angle < -90:
        rotacion = (180 + dominant_angle) + 90 #-133º
    else:
        rotacion = 90 - (180 - dominant_angle) #133º

# Calculamos el centro de la imagen para la rotación
alto, ancho = resized_image.shape[0:2]
centro = (ancho // 2, alto // 2)
angulo = rotacion
escala = 1.0

# Matriz de rotacion
matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, escala)

# Aplicamos la rotacion
imagen_rotada = cv2.warpAffine(resized_image, matriz_rotacion, (ancho, alto))

# Mostrar la imagen con las líneas detectadas
cv2.imshow('Deteccion de lineas', imagen_rotada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================
# 3) MORFOLOGÍA SOBRE TU CANNY
# =============================

# Usamos TU Canny directamente sobre la imagen rotada
canny_rot = cv2.warpAffine(bordes, matriz_rotacion, (ancho, alto))

cv2.imshow("Canny tras rotar", canny_rot)
cv2.waitKey(0)

# Closing morfológico para unir las barras en una masa
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
closed = cv2.morphologyEx(canny_rot, cv2.MORPH_CLOSE, kernel, iterations=2)

# Limpieza básica
closed = cv2.dilate(closed, None, iterations=1)
closed = cv2.erode(closed, None, iterations=1)

cv2.imshow("Closed (solo con Canny)", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================
# 4) CONTORNOS + FILTROS
# =============================

cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

if not cnts:
    raise RuntimeError("No se han encontrado contornos.")

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
barcode_contour = None

for c in cnts:
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    if w == 0 or h == 0:
        continue

    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio < 1.8:   # muy alargado
        continue

    # Filtro de orientacion
    norm_angle = angle
    if w < h:
        norm_angle = angle + 90

    if abs(norm_angle) > 40:
        continue

    barcode_contour = c
    break

if barcode_contour is None:
    raise RuntimeError("No se ha encontrado código de barras.")

debug = imagen_rotada.copy()
cv2.drawContours(debug, [barcode_contour], -1, (0, 255, 0), 2)
cv2.imshow("Contorno seleccionado", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================
# 5) RECTIFICAR ROI DEL BARCODE
# =============================

rect = cv2.minAreaRect(barcode_contour)
box = cv2.boxPoints(rect)
box = np.float32(box)

# Ordenar puntos para warp
def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

ordered_box = order_points(box)

width = int(rect[1][0])
height = int(rect[1][1])

if width < height:
    width, height = height, width

dst = np.array([
    [0, height-1],
    [width-1, height-1],
    [width-1, 0],
    [0, 0]
], dtype="float32")

M = cv2.getPerspectiveTransform(ordered_box, dst)
barcode_roi = cv2.warpPerspective(imagen_rotada, M, (width, height))

cv2.imshow("ROI rectificada", barcode_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
