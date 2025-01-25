# Python program to explain cv2.erode() method 

# importing cv2 
import cv2 

# importing numpy 
import numpy as np 

# path 
path = r'./Figure_1.png'

# Reading an image in default mode 
image = cv2.imread(path) 

# Window name in which image is displayed 
window_name = 'Image'

# Creating kernel 
kernel = np.ones((5,5), np.uint8) 

# Using cv2.erode() method 
image = cv2.blur(image, (10,10)) 

# Displaying the image 


cv2.imwrite("Output.png", image)
