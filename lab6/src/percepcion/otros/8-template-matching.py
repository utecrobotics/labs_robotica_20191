import cv2
import numpy as np
from matplotlib import pyplot as plt

I = cv2.imread('imagenes/gatos.jpg',0)
template = cv2.imread('imagenes/gato_rostro.jpg',0)
width, height = template.shape

method = cv2.TM_CCOEFF_NORMED
#method = cv2.TM_CCORR_NORMED

# Aplicar template Matching
Iscores = cv2.matchTemplate(I, template, method)
# Extraer la localizacion
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Iscores)
top_left = max_loc
bottom_right = (top_left[0]+width, top_left[1]+height)

# Dibujar un rectangulo en la parte que corresponde
cv2.rectangle(I, top_left, bottom_right, 255, 2)

plt.subplot(121), plt.imshow(Iscores, cmap = 'gray')
plt.title('Resultado de Matching'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(I, cmap = 'gray')
plt.title('Objeto detectado'), plt.xticks([]), plt.yticks([])
plt.show()

# Detectar multiples correspondencias
I = cv2.imread('imagenes/gatos.jpg',0)
threshold = 0.55
location = np.where(Iscores >= threshold)
for pt in zip(*location[::-1]):
    cv2.rectangle(I, pt, (pt[0]+width, pt[1]+height), (0,0,255), 2)
cv2.imshow('Resultado', I)
cv2.waitKey(0)
