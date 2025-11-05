import cv2

# Cargar la imagen
img = cv2.imread('img/test/Barcodes_train_010195.jpg')

# Definir los factores de escala
scale_x = 0.5
scale_y = 0.5

# Redimensionar la imagen utilizando factores de escala
resized_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

# Convertir a escala de grises
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Aplicar el detector de bordes Canny
bordes = cv2.Canny(gray, 50, 220) 

# Mostrar los bordes detectados
cv2.imshow('Bordes Canny', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()