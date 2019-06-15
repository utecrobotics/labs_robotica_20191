import numpy as np
import cv2
from matplotlib import pyplot as plt

# Cargar una imagen como gris (1 solo canal)
I = cv2.imread('imagenes/utec1.jpg', 0)
# Mostrar la imagen
cv2.imshow("Edificio de UTEC", I), cv2.waitKey(0)

# Calcular el histograma: 
#    (I, canal, 0, mask=None, histogram size, range 
hist = cv2.calcHist([I], [0], None, [256], [0,256])
# Graficar el histograma usando matplotlib
plt.plot(hist); plt.show()

# ---------------------------------------
# Ecualizacion de histograma
# ---------------------------------------

I = cv2.imread("imagenes/oscuro.jpg", 0)
Ihisteq = cv2.equalizeHist(I)

cv2.imshow("Imagen original", I)
cv2.imshow("Imagen ecualizada", Ihisteq)
cv2.waitKey(0)
