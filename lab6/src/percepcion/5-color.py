import numpy as np
import cv2

I = cv2.imread("imagenes/gatos.jpg")
cv2.imshow("Imagen original", I); cv2.waitKey(0)

# Conversion a escala de grises
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", Igray); cv2.waitKey(0)

# Conversion a HSV
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
# Extraer y mostrar solo el componente de Hue
Ihue = Ihsv[:,:,0]
cv2.imshow("Hue", Ihue); cv2.waitKey(0)

# Segmentar amarillo usando "hue"
# -------------------------------
# Limites superior e inferior para amarillo
lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([40, 255, 255])
# Mascara que selecciona pixeles en los limites anteriores
mask = cv2.inRange(Ihsv, lower_yellow, upper_yellow)
# Applicacion de la mascara a la imagen para mantener la region de interes
Iyellow = cv2.bitwise_and(I, I, mask=mask)

# Mostrar la imagen
cv2.imshow("Parte amarilla de la imagen", Iyellow)
cv2.waitKey(0)
cv2.destroyAllWindows()
