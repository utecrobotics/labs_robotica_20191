#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist



rospy.init_node("control")


vel = Twist()
vel.linear.x = 0
vel.linear.y = 0
vel.linear.z = 0
vel.angular.x = 0
vel.angular.y = 0
vel.angular.z = 0



# Envio de datos con una frecuencia de 100 Hz (100 por segundo)
rate = rospy.Rate(100)
# Bucle de ejecucion continua
while not rospy.is_shutdown():
    
    
    
    rate.sleep()
