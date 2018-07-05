# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('..\\testim\\apl.jpg',0)
edges = cv2.Canny(img, 200,400)
n_white_pix = np.sum(edges == 255)
print('Number of white pixels:', n_white_pix)


def square_it(img):
	y, x = img.shape
	if x > y:
		left = img[0:y, 0:y]
		centre = img[0:y, (x//2)-(y//2):(x//2)+(y//2)]
		right = img[0:y, x-y:x]
	else:
		left = img[0:x, 0:x]
		centre = img[(y//2)-(x//2):(y//2)+(x//2), 0:x]
		right = img[y-x:y, 0:x]
	return whites(left, centre, right)

def whites(l, c, r):
	lw = np.sum(l == 255)
	cw = np.sum(c == 255)
	rw = np.sum(r == 255)
	if lw > cw:
		if lw > rw:
			return l
		else:
			return r
	elif rw > cw:
		return r
	else:
		return c

edcr = square_it(edges)

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(edcr,cmap = 'gray')
plt.title('Cropped Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()