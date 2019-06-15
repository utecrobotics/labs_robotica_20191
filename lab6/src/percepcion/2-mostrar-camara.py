import numpy as np 
import cv2

cam = cv2.VideoCapture(0)

while (cam.isOpened()):
    # Leer los frames: retval=True si hay una imagen valida en la camara
    retval, frame = cam.read()
    # Invertir la imagen horizontalmente (de derecha a izquierda)
    frame = cv2.flip(frame, 1)
    # Si hay una imagen valida
    if retval == True:
        # Mostrar la imagen de la camara
        cv2.imshow("My camera", frame)
        # Esperar 30 segundos. Finalizar si se presiona q
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar espacio de memoria
cam.release()
# Cerrar la ventana abierta
cv2.destroyAllWindows()
