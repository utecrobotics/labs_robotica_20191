import cv2
import numpy as np

I = cv2.imread('imagenes/gato.jpg')
Nrows, Ncols, Nchannels = I.shape

# Escalamiento
Irescaled = cv2.resize(I, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Imagen escalada', Irescaled); cv2.waitKey(0)

# Traslacion
tx = 100; ty = 50
M = np.float32([[1, 0, tx],
				[0, 1, ty]])
Itransl = cv2.warpAffine(I, M, (Ncols, Nrows)) # Size is width, height
cv2.imshow('Imagen trasladada', Itransl); cv2.waitKey(0)

# Rotacion
center_rot = (Ncols/2, Nrows/2)
angle = 45
M = cv2.getRotationMatrix2D(center_rot, angle, 1)
Irot = cv2.warpAffine(I, M, (Ncols, Nrows))
cv2.imshow('Imagen rotada', Irot); cv2.waitKey(0)

cv2.destroyAllWindows()
