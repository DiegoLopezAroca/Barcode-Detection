import cv2
import imutils
import numpy as np
from pyzbar import pyzbar
import os

FIRST_DIGIT_PARITY = {
    "LLLLLL": 0,
    "LLGLGG": 1,
    "LLGGLG": 2,
    "LLGGGL": 3,
    "LGLLGG": 4,
    "LGGLLG": 5,
    "LGGGLL": 6,
    "LGLGLG": 7,
    "LGLGGL": 8,
    "LGGLGL": 9,
}

L_CODES_MAP = {
    (3, 2, 1, 1): 0,
    (2, 2, 2, 1): 1,
    (2, 1, 2, 2): 2,
    (1, 4, 1, 1): 3,
    (1, 1, 3, 2): 4,
    (1, 2, 3, 1): 5,
    (1, 1, 1, 4): 6,
    (1, 3, 1, 2): 7,
    (1, 2, 1, 3): 8,
    (3, 1, 1, 2): 9,
}

G_CODES_MAP = {
    (1, 1, 2, 3): 0,
    (1, 2, 2, 2): 1,
    (2, 2, 1, 2): 2,
    (1, 1, 4, 1): 3,
    (2, 3, 1, 1): 4,
    (1, 3, 2, 1): 5,
    (4, 1, 1, 1): 6,
    (2, 1, 3, 1): 7,
    (3, 1, 2, 1): 8,
    (2, 1, 1, 3): 9,
}

def preprocesamiento(img):
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

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # Closing morfológico para unir las barras en una masa
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Limpieza básica
    closed = cv2.dilate(closed, None, iterations=4)
    closed = cv2.erode(closed, None, iterations=4)

    # Encontrar contornos en la imagen procesada
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # Hacer un bounding box alrededor del contorno más grande
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = box.astype(int)

    # Copia para futuro recorte
    imagen_copia = img.copy()

    #Vamos a guardar la rotacion, altura y anchura para poner la imagen derecha
    angulo = rect[2]
    alto, ancho = img.shape[0:2]
    if angulo < -45:
        angulo = -(90 + angulo)
    else:
        angulo = angulo - 90
    matriz_rotacion = cv2.getRotationMatrix2D(rect[0], angulo, 1.0)
    imagen_rotada = cv2.warpAffine(img, matriz_rotacion, (ancho, alto))
    imagen_copia = cv2.warpAffine(imagen_copia, matriz_rotacion, (ancho, alto))
    pts_rotados = cv2.transform(np.array([box]), matriz_rotacion)[0]

    # Bounding box tras rotar
    xs = pts_rotados[:, 0]
    ys = pts_rotados[:, 1]

    # Cogemos las coordenadas minimas y maximas ya que sino la imagen recortada sale mal
    x_min, x_max = xs.min()-20, xs.max()+20
    y_min, y_max = ys.min()-20, ys.max()+20

    # Recortamos la imagen
    imagen_recortada = imagen_rotada[y_min:y_max, x_min:x_max]

    return imagen_recortada

def señal_rasterizada(imagen_recortada):
    gray = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0) # Filtro Gaussiano para reducir el ruido
    val, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alto, ancho = thresh_img.shape[:2]
    fila_central_idx = alto // 2
    scan_line = thresh_img[fila_central_idx, 15:ancho-15] # Esto es debido a que le hemos añadido un ancho de 20 pixles

    scan_line = scan_line.tolist()

    count = 0
    for val in scan_line:
        if val == 255:
            count += 1
        else:
            break

    scan_line = scan_line[count:]
    count = 0

    for val in scan_line[::-1]:
        if val == 255:
            count += 1
        else:
            break

    if count > 0:
        scan_line = scan_line[:-count]

    for i in range(len(scan_line)):
        if scan_line[i] == 255:
            scan_line[i] = 0
        else:
            scan_line[i] = 1

    return scan_line

def calcular_anchos(lista_pixeles):
    lista_colores = []
    lista_anchos = []
    if not lista_pixeles:
        return lista_anchos, lista_colores
    
    contador = 0
    valor_actual = lista_pixeles[0]

    for pixel in lista_pixeles:
        if pixel == valor_actual:
            contador += 1
        else:
            lista_colores.append(valor_actual)
            lista_anchos.append(contador)
            valor_actual = pixel
            contador = 1

    lista_anchos.append(contador)
    tupla = tuple(zip(lista_colores, lista_anchos))
    
    return tupla, lista_anchos

def limpiar_señal_rasterizada(tupla):
    if len(tupla) == 0:
        print("Tupla vacía")
        return 0
    
    # Verificar si ya empieza con una barra negra (sin zona blanca grande antes)
    color_primero, ancho_primero = tupla[0]
    if color_primero == 1:  # Ya empieza con negro (barra)
        print("Inicio ya limpio, empieza con barra negra")
        return 0
    
    # Buscar el inicio del código si hay zona blanca antes
    for i in range(len(tupla) - 1):
        color_zona, ancho_zona = tupla[i]
        color_barra, ancho_barra = tupla[i+1]
        
        if color_zona == 0 and color_barra == 1:
            if ancho_zona > ancho_barra * 4:
                print(f"Inicio encontrado en posición {i+1}")
                return i + 1
    
    # Si no se encontró una zona blanca grande, asumir que empieza desde el principio
    print("No se encontró zona blanca grande, usando inicio original")
    return 0

def normalizador(anchos_modulos, chunk, modulo_local):
    for p in chunk:
        m = round(p / modulo_local)
        anchos_modulos.append(max(1, int(m)))
    return anchos_modulos

def normalizador_laterales(anchos_pixeles, indice, anchos_modulos):
    for _ in range(6):
        grupo = anchos_pixeles[indice : indice + 4]
        modulo_local = sum(grupo) / 7.0
        normalizador(anchos_modulos, grupo, modulo_local)
        indice += 4
    return anchos_modulos, indice 

def normalizar_señal_adaptativa(anchos_pixeles):
    # Validación
    if len(anchos_pixeles) < 59:
        print("Error: Faltan datos en la señal rasterizada.")
        return []

    anchos_modulos = []
    indice = 3

    # Lado izquierdo
    anchos_modulos, indice = normalizador_laterales(anchos_pixeles, indice, anchos_modulos)

    # Guard bars centrales
    grupo = anchos_pixeles[indice : indice + 5]
    modulo_local = sum(grupo) / 5.0
    anchos_modulos = normalizador(anchos_modulos, grupo, modulo_local)
    indice += 5

    # Lado derecho
    anchos_modulos, indice = normalizador_laterales(anchos_pixeles, indice, anchos_modulos)
    return anchos_modulos

def decodificar(mitad, que_mitad):

    decoded_digits = []
    parity_pattern = ""

    for i in range(0, len(mitad), 4):
        chunk = tuple(mitad[i : i+4])

        while sum(chunk) > 7:
            print(f"Corrigiendo chunk: {chunk} (Suma: {sum(chunk)})")
            modificar_mod = max(chunk) - 1
            lista = list(chunk)
            lista[lista.index(max(chunk))] = modificar_mod
            chunk = tuple(lista)
            print(f"Chunk corregido: {chunk} (Suma: {sum(chunk)})")
            
        while sum(chunk) < 7:
            print(f"Corrigiendo chunk: {chunk} (Suma: {sum(chunk)})")
            modificar_mod = min(chunk) + 1
            lista = list(chunk)
            lista[lista.index(min(chunk))] = modificar_mod
            chunk = tuple(lista)
            print(f"Chunk corregido: {chunk} (Suma: {sum(chunk)})")
        
        digit = None
        code_type = None

        if chunk in L_CODES_MAP:
            digit = L_CODES_MAP[chunk]
            code_type = "L"
        elif chunk in G_CODES_MAP:
            digit = G_CODES_MAP[chunk]
            code_type = "G"
   
        if digit is not None:
            decoded_digits.append(digit)
            if que_mitad == 0:
                parity_pattern += code_type

    if parity_pattern in FIRST_DIGIT_PARITY and que_mitad == 0:
        decoded_digits.insert(0, FIRST_DIGIT_PARITY[parity_pattern])
    
    return decoded_digits

image_list = []
for filename in os.listdir('images'):
    if '.jpg' in filename:
         image_list.append(filename)

for imagen in image_list:
    img = cv2.imread('images/' + imagen, cv2.IMREAD_COLOR)
    imagen_recortada = preprocesamiento(img)
    scan_line = señal_rasterizada(imagen_recortada)
    tupla_señal_rle, anchos_en_pixeles = calcular_anchos(scan_line)
    inicio_codigo = limpiar_señal_rasterizada(tupla_señal_rle)
    anchos_en_pixeles = anchos_en_pixeles[inicio_codigo:]
    anchos_modulos = normalizar_señal_adaptativa(anchos_en_pixeles)

    fin_izq = 24
    inicio_der = fin_izq + 5

    primera_mitad = anchos_modulos[:fin_izq]
    segunda_mitad = anchos_modulos[inicio_der:]

    if len(primera_mitad) != 24 or len(segunda_mitad) != 24:
        print("Revisar la detección de bordes.")

    decoded_digits = decodificar(primera_mitad, 0)
    decoded_digits += (decodificar(segunda_mitad, 1))

    print(decoded_digits)