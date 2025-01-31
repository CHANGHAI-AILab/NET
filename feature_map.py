import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('C:\\Users\\timchen91\\Desktop\\1.png', 0)
colormap = plt.get_cmap('inferno')
heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
print("kk")
cv2.imshow('image', image)
cv2.imshow('heatmap', heatmap)
cv2.waitKey()