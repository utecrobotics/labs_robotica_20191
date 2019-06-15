import numpy as np
import cv2
from matplotlib import pyplot as plt

I1 = cv2.imread('imagenes/box.jpg',0)
I2 = cv2.imread('imagenes/box_in_scene.jpg',0)

# ==================================
#     Usando ORB y Fuerza bruta
# ==================================

# Inicializar ORB
orb = cv2.ORB_create()
# Encontrar keypoints y descriptores con ORB
keypoints1, descriptors1 = orb.detectAndCompute(I1, None)
keypoints2, descriptors2 = orb.detectAndCompute(I2, None)

# Crear un elemento que hace matching usando fuerza bruta
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Matching de descriptores
matches = bfmatcher.match(descriptors1, descriptors2)
# Ordenar las correspondencias en orden de distancias
matches = sorted(matches, key = lambda x:x.distance)

# Dibujar las 10 primeras correspondencias
I3 = cv2.drawMatches(I1, keypoints1, I2, keypoints2, matches[:10], None, flags=2)
# Mostrar el resultado usando matplotlib
plt.imshow(I3), plt.show()


# ==================================
#    Using SIFT and FLANN matching
# ==================================

# Iniciar SIFT
sift = cv2.xfeatures2d.SIFT_create()
# Encontrar los keypoints y descriptores con SIFT
keypoints1, descriptors1 = sift.detectAndCompute(I1, None)
keypoints2, descriptors2 = sift.detectAndCompute(I2, None)

# Parametros FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flannmatcher = cv2.FlannBasedMatcher(index_params,search_params)
# Match de descriptors
matches = flannmatcher.knnMatch(descriptors1, descriptors2, k=2)

# Se requiere dibujar solo matches adecuados: crear una mascara
matchesMask = [[0,0] for i in xrange(len(matches))]

# Test de la razon
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
I3 = cv2.drawMatchesKnn(I1, keypoints1, I2, keypoints2, matches, None,
                        **draw_params)
plt.imshow(I3,), plt.show()
