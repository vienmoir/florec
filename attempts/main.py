# coding: utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flonet import FloNet
from cropim import CropIm

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

print('[info] imported everything, yay')

args = {
    "dataset": "D:\\Uni\\TU\\DL\\project\\florec\\dataset",
    "model": "D:\\Uni\\TU\\DL\\project\\florec\\florec.model",
    "labelbin": "D:\\Uni\\TU\\DL\\project\\florec\\lb.pickle",
    "plot": "D:\\Uni\\TU\\DL\\project\\florec\\plot.png"
}

# the fun part
EPOCHS = 25
INIT_LR = 1e-3 # initial learning rate (Adam will handle it later)
BS = 16 # batch  size
IMAGE_DIMS = (120, 120, 3)

#initialize data labels
data = []
labels = []

print("[info] loading paths...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

print('[info] loading images...')
for imagePath in imagePaths:
    image = CropIm(imagePath)
   # print(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
print('[info] images collected')

data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)
print("[info] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.1, 
                                                  random_state = 42)

aug = ImageDataGenerator(rotation_range = 25, width_shift_range = 0.1,
                       height_shift_range = 0.1, shear_range = 0.2, 
                        zoom_range = 0.2, horizontal_flip = True, 
                         fill_mode = "nearest")

print("[info] compiling model...")
model = FloNet.build(width = IMAGE_DIMS[1], height = IMAGE_DIMS[0],
                   depth = IMAGE_DIMS[2], classes = len(lb.classes_))
opt = Adam(lr= INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(loss = "categorical_crossentropy", optimizer = opt,
             metrics = ["accuracy"])


print("[info] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = BS),
                       validation_data = (testX, testY),
                       steps_per_epoch = len(trainX) // BS,
                       epochs = EPOCHS, verbose = 2)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()
plt.savefig(args["plot"])