import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread(r'C:\Users\arhyr\Desktop\test.jpg')
print(type(image), image.shape)

img_copy = np.copy(image)

left_bottom = [0, 409]
right_bottom = [700, 409]
apex = [350, 200]


fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX, YY = np.meshgrid(np.arange(0, img_copy.shape[1]), np.arange(0, img_copy.shape[0]))
thresholds = (YY < XX * fit_left[0] + fit_left[1]) | (YY < XX * fit_right[0] + fit_right[1]) | (
            YY > XX * fit_bottom[0] + fit_bottom[1])
img_copy[thresholds] = [0, 0, 0]

r_threshold = 200
g_threshold = 200
b_threshold = 200
rgb_threshold = [r_threshold, g_threshold, b_threshold]

thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (
            image[:, :, 2] < rgb_threshold[2])

img_copy[thresholds] = [0, 0, 0]

plt.imshow(img_copy)
plt.show()
