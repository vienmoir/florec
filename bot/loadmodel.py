# coding: utf-8
from keras.models import load_model
import numpy as np
import pickle
#import os

def LoadModel():
	print("[info] loading network...")
	model = load_model('..\\model\\3c98a.model')
	lb = pickle.loads(open('..\\model\\3c98a.pickle', "rb").read())
	print("model loaded")
	return model, lb