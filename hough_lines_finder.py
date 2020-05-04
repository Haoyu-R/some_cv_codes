import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image = mpimg.imread(r'C:\Users\arhyr\Desktop\test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

conjuction_img = np.zeros_like(edges)
verticles = np.array([[(100, 409), (200, 209), (500, 209), (600, 409)]], dtype=np.int32)
cv2.fillPoly(conjuction_img, verticles, 1)
masked_edge = cv2.bitwise_and(edges, conjuction_img)

rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 5
max_line_gap = 1
blank = np.copy(image)*0

lines = cv2.HoughLinesP(masked_edge, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(blank, (x1, y1), (x2, y2), (255, 0, 0), 2)

color_img = np.dstack((edges, edges, edges))

combi_img = cv2.addWeighted(blank, 0.8, color_img, 1, 0)
plt.imshow(combi_img)
plt.show()
