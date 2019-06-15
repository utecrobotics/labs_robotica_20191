import cv2
from matplotlib import pyplot as plt

I = cv2.imread('imagenes/utec2.jpg', 0)

# Filtro de Sobel
Isobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)
Isobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)
# Filtro Laplaciano
Ilaplacian = cv2.Laplacian(I, cv2.CV_64F)

plt.subplot(2,2,1), plt.imshow(I, cmap='gray')
plt.title('Imagen original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(Ilaplacian, cmap='gray')
plt.title('Laplaciano'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(Isobelx, cmap='gray')
plt.title('Filtro de Sobel en X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(Isobely, cmap='gray')
plt.title('Filtro de Sobel en Y'), plt.xticks([]), plt.yticks([])
plt.show()

# Deteccion de bordes usando Canny
I = cv2.imread('imagenes/utec3.jpg', 0)
Iedges = cv2.Canny(I, 150, 200)

plt.subplot(121), plt.imshow(I, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(Iedges, cmap = 'gray')
plt.title('Edge Image (canny)'), plt.xticks([]), plt.yticks([])
plt.show()
