import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

import numpy as np

image = mpimg.imread(r'C:\Users\arhyr\Desktop\test.jpg')
print(type(image), image.shape)

low_blue = np.array([0, 0, 50])
upper_blue = np.array([100, 100, 255])

mask = cv2.inRange(image, low_blue, upper_blue)

# plt.imshow(mask, cmap='gray')
# new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(new_img)
# plt.imshow(image)
# new_img = np.copy(image)
# new_img[mask != 0] = [255, 255, 255]
# plt.imshow(new_img)
image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20, 10))
ax1.imshow(image[:, :, 0], cmap='gray')
ax2.imshow(image[:,:,1], cmap='gray')
ax3.imshow(image[:,:,2], cmap='gray')

plt.show()