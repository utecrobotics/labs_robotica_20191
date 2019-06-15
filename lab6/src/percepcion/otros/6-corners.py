import cv2
import numpy as np

I = cv2.imread('imagenes/bloques.jpg',0)

# Corners de Harris
neighborhood = 2  # Tamano del vecindario
apperture = 3     # Tamano de la apertura
alpha = 0.04      # Parametro alpha (o k)
score = cv2.cornerHarris(I, neighborhood, apperture, alpha)
# Dilatar (solo para mostrar mejor las esquinas)
score = cv2.dilate(score,None)

# Mostrar los corners mayores que 0.01*max en la imagen (en azul)
Ibgr = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
Ibgr[score > 0.01*score.max()] = [255, 0, 0]

cv2.imshow('Harris Corners', Ibgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Parametros para Shi-Tomasi (goodFeaturesToTrack)
numcorners = 25  # Numero de mejores corners a mantener
quality = 0.01   # Rechazar debajo de este valor de calidad
mindist = 10     # Distancia Euclideana minima entre esquinas
# Deteccion de esquinas (corners)
corners = cv2.goodFeaturesToTrack(I, numcorners, quality, mindist)
corners = np.int0(corners)

# Dibujar circulos alrededor de las esquinas (corners)
Ibgr = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
for k in corners:
    x,y = k.ravel()
    cv2.circle(Ibgr, (x,y), 3, 255, -1)

cv2.imshow('Shi-Tomasi', Ibgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


