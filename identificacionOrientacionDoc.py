from argparse import ArgumentParser

import cv2
import imutils
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cargamos la imagen y la mostramos en pantalla.
image = cv2.imread(r"C:\Users\daft1\OneDrive\Documentos\red Neuronal Generativa\GFPGAN\results\restored_imgs\img_7.jpg")
cv2.imshow('Imagen', image)
cv2.waitKey(0)

# Como OpenCV carga las imágenes en formato BGR, debemos transformarla a RGB para que sea compatible con Tesseract.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Usamos Tesseract para extraer la orientación del texto en la imagen.
results = pytesseract.image_to_osd(image, output_type=Output.DICT, lang="spa")

if results['rotate'] != 0:
    # Rotamos la imagen
    rotated = imutils.rotate_bound(image, angle=results['rotate'])
    rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
    cv2.imshow('Rotada', rotated)
    cv2.waitKey(0)
    # Destruimos las ventanas creadas.
    cv2.destroyAllWindows()