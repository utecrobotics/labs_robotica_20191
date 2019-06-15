#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class Camera(object):
    def __init__(self):
        topic = "/camera/rgb/image_raw/compressed"
        self.image_sub = rospy.Subscriber(topic, CompressedImage, self.callback)
        self.cimage = CompressedImage()
        self.bridge = CvBridge()

    def callback(self, msg):
        self.cimage = msg

    def get_cvimage(self):
        # Convierte la imagen de topico a imagen de opencv
        return self.bridge.compressed_imgmsg_to_cv2(self.cimage)        


# --------------------------------------------------------------------
#       Programa Principal 
# --------------------------------------------------------------------

rospy.init_node("nodo_imagen")

cam = Camera()
rospy.sleep(1)

# Ejecutar el bucle 30 veces por segundo
rate = rospy.Rate(30)

while not rospy.is_shutdown():

    # La imagen I esta en formato de OpenCV
    I = cam.get_cvimage()

    cv2.imshow("Ventana", I)
    cv2.waitKey(1)
   
    rate.sleep()


    
cv2.destroyAllWindows()
