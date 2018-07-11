# coding: utf-8
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from cropim import CropIm
import numpy as np
#import argparse
import imutils
import pickle
import cv2
import os

# load the image
path = '..\\testim\\tilia_cordata\\tilia_cordata_164.jpg'
image = CropIm(path)
output = cv2.imread(path,3)
 
# pre-process the image for classification
image = cv2.resize(image, (120, 120))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# load the trained convolutional neural network and the label
# binarizer
print("[info] loading network...")
model = load_model('..\\model\\5c85a.model')
lb = pickle.loads(open('..\\model\\5c85a.pickle', "rb").read())
 
# classify the input image
print("[info] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
filename = path[path.rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"
 
# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (158, 209, 48), 2)
 
# show the output image
print("[INFO] {}".format(label))
cv2.imwrite("..\\results\\classify\\aesc1.png", output)
cv2.imshow("Output", output)
cv2.waitKey(0)