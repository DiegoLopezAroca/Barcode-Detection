import numpy as np
import cv2

# Cargar la imagen en color
image = cv2.imread('img/test/barcode_detector_train_025407.jpg', cv2.IMREAD_COLOR)

# Comprobar que la imagen se ha leído bien
if image is None:
    raise FileNotFoundError("No se ha podido cargar la imagen. Revisa la ruta: 'img/test/barcode_detector_train_025407.jpg'")
# Redimensionar imagen
image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gradiente en X e Y
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# Gradiente combinado
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

cv2.imshow("gradient-sub", cv2.resize(gradient, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

# Suavizado
blurred = cv2.blur(gradient, (3, 3))

# Umbralización (puedes ajustar el 225 si hace falta)
_, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

cv2.imshow("threshed", cv2.resize(thresh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

# Cierre morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow("morphology", cv2.resize(closed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

# Erosiones y dilataciones
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

cv2.imshow("erode/dilate", cv2.resize(closed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

# Buscar contornos (versión OpenCV 4.x)
# Si usas OpenCV 3.x, la firma es: image, cnts, hierarchy = cv2.findContours(...)
cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not cnts:
    print("No se han encontrado contornos.")
else:
    # Ordenar por área y coger el más grande (y opcionalmente el segundo)
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Primer contorno (el de mayor área)
    c = cnts_sorted[0]
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # np.int0 está deprecado

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # Si quieres dibujar también el segundo contorno, sólo si existe
    if len(cnts_sorted) > 1:
        c1 = cnts_sorted[1]
        rect1 = cv2.minAreaRect(c1)
        box1 = cv2.boxPoints(rect1)
        box1 = np.intp(box1)
        cv2.drawContours(image, [box1], -1, (0, 0, 255), 3)  # en rojo para distinguir

# Redimensionar para visualizar
image_vis = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

cv2.imshow("Image", image_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
