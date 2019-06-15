import numpy as np
import cv2
from matplotlib import pyplot as plt

I = cv2.imread('imagenes/utec1.jpg')
# Creacion del filtro
filter = np.ones((5,5), np.float32)/25
# Aplicacion del filtro (-1: mantener la misma profundidad)
Ifilt = cv2.filter2D(I, -1, filter)

plt.subplot(121),
plt.imshow(cv2.cvtColor(I,cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(Ifilt, cv2.COLOR_BGR2RGB))
plt.title('Suavizado'), plt.xticks([]), plt.yticks([])
plt.show()

# Otras formas de media
Iblur = cv2.blur(I, (5,5))
Iblurg = cv2.GaussianBlur(I, (5,5), 0)
Imedian = cv2.medianBlur(I, 5)

cv2.imshow('Imagen usando media', Iblur)
cv2.imshow('Imagen con filtro Gaussiano', Iblurg)
cv2.imshow('Imagen con filtro de mediana', Imedian)

cv2.waitKey(0)
cv2.destroyAllWindows()
