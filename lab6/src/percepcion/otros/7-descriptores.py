import cv2
import numpy as np

# Lectura de la imagen en formato escala de grises
I = cv2.imread('imagenes/bloques.jpg', 0)

# =================
#  Descriptor SIFT
# =================
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(I, None)
keypoints, descriptors = sift.detectAndCompute(I, None)

# Dibujar keypoints
Isift = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Isift,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT', Isift); cv2.waitKey(0)
cv2.destroyAllWindows()


# =================
#  Descriptor SURF
# =================
hessian_threshold = 4000
surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
keypoints, descriptors = surf.detectAndCompute(I, None)

# Dibujar keypoints
Isurf = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Isurf, (255,0,0), 4)

cv2.imshow('SURF', Isurf); cv2.waitKey(0)
cv2.destroyAllWindows()


# =================
#  Descriptor FAST
# =================
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(I, None)

# Dibujar keypoints
Ifast = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Ifast, color=(255,0,0))

cv2.imshow('FAST', Ifast); cv2.waitKey(0)
cv2.destroyAllWindows()


# ==================
#  Descriptor BRIEF
# ==================
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# Encontrar keypoints con STAR
keypoints = star.detect(I, None)
# Calcular los descriptores con BRIEF
keypoints, descriptors = brief.compute(I, keypoints)

# Dibujar keypoints
Ibrief = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Ibrief, color=(0,255,0), flags=0)

cv2.imshow('BRIEF', Ibrief); cv2.waitKey(0)
cv2.destroyAllWindows()


# ================
#  Descriptor ORB
# ================
orb = cv2.ORB_create()
keypoints = orb.detect(I, None)
keypoints, descriptors = orb.compute(I, keypoints)

# Dibujar keypoints
Iorb = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
cv2.drawKeypoints(I, keypoints, Iorb, color=(0,255,0), flags=0)

cv2.imshow('ORB', Iorb); cv2.waitKey(0)
cv2.destroyAllWindows()
