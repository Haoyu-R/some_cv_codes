import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread(r'C:\Users\arhyr\Desktop\test.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_img, cmap='gray')

blur_gray = cv2.GaussianBlur(gray_img, (7, 7), 0)

edges = cv2.Canny(gray_img, 50, 150)

plt.imshow(edges, cmap='Greys_r')
plt.show()