import cv2
import numpy as np

I = cv2.imread('imagenes/utec1.jpg', 0)
# Valor de umbral (threshold)
thval = 150
retval, Ibw = cv2.threshold(I, thval, 255, cv2.THRESH_BINARY)

cv2.imshow('Imagen original',I)
cv2.imshow('Imagen con threshold', Ibw)

cv2.waitKey(0)
cv2.destroyAllWindows()
