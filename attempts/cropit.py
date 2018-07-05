# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:\\Uni\\TU\\DL\\project\\florec\\testim\\apl.jpg',3)
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
edges = cv2.Canny(img, 200,400)
n_white_pix = np.sum(edges == 255)
print('Number of white pixels:', n_white_pix)

def square_it(grim, orim):
	y, x = grim.shape
	if x > y:
		lw = np.sum(grim[0:y, 0:y] == 255)
		cw = np.sum(grim[0:y, (x//2)-(y//2):(x//2)+(y//2)] == 255)
		rw = np.sum(grim[0:y, x-y:x] == 255)
	else:
		lw = np.sum(grim[0:x, 0:x] == 255)
		cw = np.sum(grim[(y//2)-(x//2):(y//2)+(x//2), 0:x] == 255)
		rw = np.sum(grim[y-x:y, 0:x] == 255)
	if lw > rw:
		if lw > cw:
			if x > y:
				return grim[0:y, 0:y], orim[0:y, 0:y]
			else:
				return grim[0:x, 0:x], orim[0:x, 0:x]
		else:
			if x > y:
				return grim[0:y, (x//2)-(y//2):(x//2)+(y//2)], orim[0:y, (x//2)-(y//2):(x//2)+(y//2)]
			else:
				return grim[(y//2)-(x//2):(y//2)+(x//2), 0:x], orim[(y//2)-(x//2):(y//2)+(x//2), 0:x]
	elif rw > cw:
		if x > y:
			return grim[0:y, x-y:x], orim[0:y, x-y:x]
		else:
			return grim[y-x:y, 0:x], orim[y-x:y, 0:x]
	else:
		if x > y:
			return grim[0:y, (x//2)-(y//2):(x//2)+(y//2)], orim[0:y, (x//2)-(y//2):(x//2)+(y//2)]
		else:
			return grim[(y//2)-(x//2):(y//2)+(x//2), 0:x], orim[(y//2)-(x//2):(y//2)+(x//2), 0:x]

edcr, orcr = square_it(edges, img)

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(edcr,cmap = 'gray')
plt.title('Cropped Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(orcr)
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])

plt.show()