import os
from pyzbar import pyzbar
import numpy as np
import cv2

# -----------------------------------------------------------
# PREPROCESS
# -----------------------------------------------------------
def preprocess(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    # Resize
    image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gradientes X e Y
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Resta de gradientes
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # Suavizado
    blurred = cv2.blur(gradient, (3, 3))

    # Umbralizado
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    return thresh


# -----------------------------------------------------------
# BARCODE READER
# -----------------------------------------------------------
def barcode(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, 0)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    # Detect barcodes
    barcodes = pyzbar.decode(image)

    if not barcodes:
        print("No se detectó ningún código de barras")
        return

    # Extract results
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        print(f'Format: {barcode.type}, Data: {barcode_data}')
        print('---------------------------------')


# -----------------------------------------------------------
# MAIN (SIN ARGPARSE)
# -----------------------------------------------------------
if __name__ == "__main__":
    image_path = "img/test/barcode_detector_test_025565.jpg"  # <--- CAMBIA AQUÍ

    processed = preprocess(image_path)

    # Si quieres ver el preprocess:
    # cv2.imshow("preprocess", processed)
    # cv2.waitKey(0)

    barcode(image_path)
