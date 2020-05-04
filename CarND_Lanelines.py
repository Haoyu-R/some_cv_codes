import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
import os
from moviepy.editor import VideoFileClip


def find_lane(image):
    # To gray, gaussianBlur and Canny edge
    # image = mpimg.imread(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blur_img, 100, 200)

    # Define region of interest
    vertices = np.array([[(65, 539), (500, 300), (897, 539)]], dtype=np.int32)
    conjunction_img = np.zeros_like(edges)
    cv2.fillPoly(conjunction_img, vertices, 1)
    masked_edges = cv2.bitwise_and(edges, conjunction_img)

    # Define color of interest
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 155], dtype=np.uint8)
    upper_white = np.array([255, 100, 255], dtype=np.uint8)
    lower_blue = np.array([110, 100, 100], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    masked_color = mask_white + mask_blue + mask_yellow

    masked_img = cv2.bitwise_and(masked_edges, masked_color)

    # Define parameter for Hough lane search
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_length = 10
    max_line_gap = 5

    lines = cv2.HoughLinesP(masked_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is None:
        return image
    # Draw lanes
    blank = np.copy(image) * 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((y1-y2)/(x1-x2) < -0.5) or ((y1-y2)/(x1-x2) > 0.5):
                cv2.line(blank, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # colored_edges = np.dstack((edges, edges, edges))

    # Add detected lane to original img
    combi_img = cv2.addWeighted(blank, 0.8, image, 1, 0)
    return combi_img


# for file in os.listdir(r'C:\Users\arhyr\Desktop\self_driving_car\CarND-LaneLines-P1\test_images'):
#     img = find_lane(r'C:\Users\arhyr\Desktop\self_driving_car\CarND-LaneLines-P1\test_images\\'+file)
#     plt.imshow(img)
#     plt.show()

clip = VideoFileClip(r'C:\Users\arhyr\Desktop\self_driving_car\CarND-LaneLines-P1\test_videos\challenge.mp4')
new_clip = clip.fl_image(find_lane)
new_clip.write_videofile(r'C:\Users\arhyr\Desktop\self_driving_car\CarND-LaneLines-P1\test_videos_output\challenge.mp4', audio=False)





