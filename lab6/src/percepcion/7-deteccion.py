import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

I1 = cv2.imread('imagenes/box.jpg',0)
I2 = cv2.imread('imagenes/box_in_scene.jpg',0)

# Detector SIFT
sift = cv2.xfeatures2d.SIFT_create()
# Keypoints y descriptores con SIFT
keypoints1, descriptors1 = sift.detectAndCompute(I1, None)
keypoints2, descriptors2 = sift.detectAndCompute(I2, None)

# Matcher FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# Match de los descriptores de ambas imagenes
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Almacenar las mejores correspondencias (matches)
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Encontrar la mejor posicion y orientacion
I2 = cv2.imread('imagenes/box_in_scene.jpg')
if len(good)>MIN_MATCH_COUNT:
    source_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    destin_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(source_pts, destin_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    height, width = I1.shape
    points = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
    destin = cv2.perspectiveTransform(points, M)

    I2 = cv2.polylines(I2, [np.int32(destin)], True, (255,0,0), 3, cv2.LINE_AA)

else:
    print "No hay suficientes correspondencias"
    matchesMask = None

draw_params = dict(matchColor = (0,255,0),    # Matches en verde
                   singlePointColor = None,
                   matchesMask = matchesMask, # Dibujar solo inliers
                   flags = 2)

I3 = cv2.drawMatches(I1, keypoints1, I2, keypoints2, good, None, **draw_params)
plt.imshow(I3, 'gray'), plt.show()
