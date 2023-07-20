#!/usr/bin/env python
# coding: utf-8

# In[67]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import pytesseract

# Load an image using cv2.imread()
img = cv2.imread(r'D:\Python\MV_viewer_images\Pic_2023_01_20_090513_10.bmp')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # adjust contrast
# gray = cv2.equalizeHist(img_rgb)


#1


median = cv2.medianBlur(img_rgb, 3)



# gaussian blur
gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 0)



# morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilation = cv2.dilate(img_rgb, kernel)
erosion = cv2.erode(img_rgb, kernel)

# Non-local Means
# nlm = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)


# plt.imshow(median)
# plt.show()

# plt.imshow(gaussian)
# plt.show()

# plt.imshow(dilation)
# plt.show()

# plt.imshow(erosion)
# plt.show()

# Define the crop region
region_width = 920
region_height = 300
region_center_x = 1250
region_center_y = 700
# region_width = 200
# region_height = 160
# region_center_x = 1680
# region_center_y = 1150


# Crop the image
cropped_img = cv2.getRectSubPix(gaussian, (region_width, region_height), (region_center_x, region_center_y))



# plt.imshow(cropped_img)
# plt.show()


#2

# Define the kernel for the morphological transformation
kernel = np.ones((10,10), np.uint8)



# Apply erosion to the image
erosion = cv2.erode(cropped_img, kernel, iterations=1)

# Apply dilation to the image
dilation = cv2.dilate(cropped_img, kernel, iterations=1)

# Apply an opening transformation to the image
opening = cv2.morphologyEx(cropped_img, cv2.MORPH_OPEN, kernel)

# Apply a closing transformation to the image
closing = cv2.morphologyEx(cropped_img, cv2.MORPH_CLOSE, kernel)


# plt.imshow(cropped_img)
# plt.show()

# plt.imshow(erosion)
# plt.show()

# plt.imshow(dilation)
# plt.show()

# plt.imshow(opening)
# plt.show()

# plt.imshow(closing)
# plt.show()

# Apply a binary threshold to the image
thresh = cv2.threshold(cropped_img, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

plt.imshow(thresh,cmap='gray')
plt.show()

#5
# Run OCR on the image using pytesseract.image_to_string()
text = pytesseract.image_to_string(thresh, config='--psm 6')

# Print the extracted text
print(text)


# In[ ]:





# In[53]:


########### TRIAL############
import cv2
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import pytesseract

# Load an image using cv2.imread()
img = cv2.imread(r'D:\Python\MV_viewer_images\Pic_2023_01_20_090513_10.bmp')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
plt.imshow(img_rgb, cmap='gray')
plt.show()

# Define the crop region
region_width = 920
region_height = 300
region_center_x = 1250
region_center_y = 700
# region_width = 200
# region_height = 160
# region_center_x = 1680
# region_center_y = 1150


# Crop the image
cropped_img = cv2.getRectSubPix(img_rgb, (region_width, region_height), (region_center_x, region_center_y))

  
plt.imshow(cropped_img, cmap='gray')
plt.show()


# In[60]:


# Apply thresholding to convert the image to binary
thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  
plt.imshow(thresh, cmap='gray')
plt.show()

#5
# Run OCR on the image using pytesseract.image_to_string()
text = pytesseract.image_to_string(binary, config='--psm 6')

# Print the extracted text
print(text)
  


# In[31]:


# Find contours in the binary image
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
print(len(cnts))



# Iterate through the contours
for c in cnts:
    # Get the rectangle bounding the contour
    x, y, w, h = cv2.boundingRect(c)

    # Extract the ROI from the image
    roi = img[y:y+h, x:x+w]

    # OCR the ROI
    text = pytesseract.image_to_string(roi)

    print(text)


# In[ ]:




