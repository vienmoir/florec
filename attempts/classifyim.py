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

def Classify(image, model, lb):
	image = cv2.resize(image, (120, 120))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	print("[info] classifying image...")
	proba = model.predict(image)[0]
	print("[info] classified")
	idx = np.argmax(proba)
	print("done")
	label = lb.classes_[idx]
	print("alright")
	print(label)
	prob = proba[idx] * 100
	return label, prob