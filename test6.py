from visual import *
#from shapefile import *
from Polygon import *
import numpy as np
import wx
from robot_functions import *
#from scene_functions import *

c = (1,1,0)
c2 = color.rgb_to_hsv(c)	# convert RGB to HSV
print(c2)	# (0.16667, 1, 1)
c3 = color.hsv_to_rgb(c2)	# convert back to RGB
print(c3)	# (1, 1, 0)



len_arm1 = 8
len_arm2 = 6
len_gripper = 2
len_base = 2

R = Robot()
R._box(1,0,color.red)
R._box(4,6,color.red)
R._box(4,6,color.blue)
R._pyramid(4,6,color.green)
R._pyramid(2,2,color.yellow)
R._cylinder(4,1,color.cyan)
R._sphere(3,5,color.red)

save = 0
#save = 'save'		#saving image

a1,a2,a3 = R.inverse_kinematics(4,6,0,'pick')
R.rotate_robot(-a1,-a2,-a3,save)
R.rotate_robot_with_object(0,-3*np.pi/4,3*np.pi/4,save)
a1,a2,a3 = R.inverse_kinematics(0,7,0,'put')
#a1,a2,a3 = R.inverse_kinematics(0,0,0,'put')
R.rotate_robot_with_object(-a1,-a2,-a3,save)
R.rotate_robot(0,-3*np.pi/4,3*np.pi/4,save)

"""
for i in range(30):
	rate(10000)
	a0 = -i*np.pi/180.0
	a1 = -i*np.pi/180.0
	a2 = -i*np.pi/180.0
	R.rotate_robot(a0,a1,a2)
"""
