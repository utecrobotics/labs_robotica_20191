import numpy as np
import cv2

I = cv2.imread('imagenes/formas.png', 0)

# Elemento estructurante
se = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
# Otras alternativas de elemento estructurante:
# cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

# Operaciones morfologicas basicas
Ierosion  = cv2.erode(I, se, iterations=1)
Idilation = cv2.dilate(I, se, iterations=1)
Iopening  = cv2.morphologyEx(I, cv2.MORPH_OPEN, se)
Iclosing  = cv2.morphologyEx(I, cv2.MORPH_CLOSE, se)

cv2.imshow('Imagen original', I)
cv2.imshow('Imagen erosionada', Ierosion)
cv2.imshow('Imagen dilatada', Idilation)
cv2.imshow('Imagen con apertura', Iopening)
cv2.imshow('Imagen con cierre', Iclosing)
cv2.waitKey(0)


# Diferencia entre dilatacion y erosion
if (False):
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gradient = cv2.morphologyEx(I, cv2.MORPH_GRADIENT, se2)
    cv2.imshow('Gradiente', gradient)
    cv2.waitKey(0)

cv2.destroyAllWindows()
