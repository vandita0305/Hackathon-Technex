#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg",frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xff ==ord('q'):
        break
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()


# Load the image of the person's body and the clothing item
body_img = cv2.imread('NewPicture.jpg')
cloth_img = cv2.imread('tshirt.jpg')

# Resize the clothing item  to fit the person's body
cloth_resized = cv2.resize(cloth_img, (body_img.shape[1], body_img.shape[0]))

# Convert the images to grayscale
body_gray = cv2.cvtColor(body_img, cv2.COLOR_BGR2GRAY)
cloth_gray = cv2.cvtColor(cloth_resized, cv2.COLOR_BGR2GRAY)

# Threshold the clothing item image to create a mask
_, cloth_mask = cv2.threshold(cloth_gray, 10, 255, cv2.THRESH_BINARY)

# Invert the mask to create a transparent area around the clothing item
cloth_mask_inv = cv2.bitwise_not(cloth_mask)
cloth_transparent = cv2.merge((cloth_gray, cloth_mask_inv, cloth_mask_inv))

# Apply the mask to the clothing item and combine it with the person's body
cloth_masked = cv2.bitwise_and(cloth_resized, cloth_transparent)
result = cv2.add(body_img, cloth_masked)

# Show the result
cv2.imshow('Result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




