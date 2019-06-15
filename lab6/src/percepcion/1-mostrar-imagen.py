import numpy as np
import cv2

I = cv2.imread("imagenes/gato.jpg")


# ===============================
# Mostrar la imagen usando opencv
# ===============================
cv2.imshow("Ventana del gato", I)
# Esperar hasta que cualquier tecla sea presionada
cv2.waitKey(0)
# Cerrar las ventanas abiertas
cv2.destroyAllWindows()


# ===================================
# Mostrar la imagen usando matplotlib
# ===================================
from matplotlib import pyplot as plt
# Convertir BGR (opencv) a RGB
Irgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
plt.imshow(Irgb)
# Eliminar los ejes y mostrar
plt.xticks([]), plt.yticks([]), plt.show()
