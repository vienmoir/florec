# coding: utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from newnet import FloNet
from cropim import CropIm

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
#import argparse
import random
import pickle
import cv2
import os

# confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

print('[info] imported everything, yay')

args = {
    "dataset": "D:\\Uni\\bac\\lvs\\project\\leaves",
    "model": "..\\leaves.model",
    "labelbin": "..\\leaves.pickle",
    "lplot": "..\\ll.png",
    "aplot": "..\\al.png",
    "confm_train": "..\\confl_train.png",
    "confm_test": "..\\confl_test.png",
    "confm_all": "..\\confl_all.png",
    "confm_normal": "..\\confbark_normal.png"
}

# the fun part
EPOCHS = 10
INIT_LR = 1e-3 # initial learning rate (Adam will handle it later)
BS = 16 # batch  size
IMAGE_DIMS = 128
#initialize data labels
data = []
labels = []

print("[info] loading paths...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

print('[info] loading images...')
for imagePath in imagePaths:
    img, cnt, coord = (imagePath)
    if type(img) != str:
        print("ok!")
        image = 
    # image = cv2.imread(imagePath,3)
    # print(imagePath)
    # image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # image = img_to_array(image)
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

aug = ImageDataGenerator(rotation_range=360, horizontal_flip = True, vertical_flip = True,
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
#model.save_weights('first_try.h5')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="test_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
plt.savefig(args["lplot"])

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="test_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
plt.savefig(args["aplot"])

def plot_confusion_matrix(cm, classes, save_path, normalize = False, title = "Confusion matrix",
cmap = plt.get_cmap('Purples')):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting mormile = true
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]).round(2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    print(cm)

    tresh =  cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment = "center",
                color = "white" if cm[i, j] > tresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path)

# calculate confusion matrix

## generator = aug.flow(testX, testY, batch_size = BS)

# y_true  = model.targets[0]
# print(y_true.axis(1))
# y_pred = model.outputs[0]
# print(y_pred.axis(1))
# cm = confusion_matrix(np.array(y_true), np.array(y_pred))
# predY = model.predict_generator(generator, verbose = 2)
predY = model.predict(testX, verbose = 2)
cm = confusion_matrix(testY.argmax(axis=1), predY.argmax(axis=1))
cm_plot_labels = lb.classes_
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_normal"], normalize= True, title="Confusion Matrix")

predY = model.predict(testX, verbose = 2)
cm = confusion_matrix(testY.argmax(axis=1), predY.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_test"], title="Confusion Matrix Test")

predY = model.predict(trainX, verbose = 2)
cm = confusion_matrix(trainY.argmax(axis=1), predY.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_train"], title="Confusion Matrix Train")


predY = model.predict(np.vstack((trainX, testX)), verbose = 2)
cm = confusion_matrix(np.vstack((trainY, testY)).argmax(axis=1), predY.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_all"], title="Confusion Matrix All")

# calculate average loss and accuracy for the last 5 training and test data

loss = 0
val_loss = 0
acc = 0
val_acc = 0

log_count = float(len(model_metrics))

print(model_metrics)

for log in model_metrics:
    loss += log['loss']
    val_loss += log['val_loss']
    acc += log['acc']
    val_acc += log['val_acc']

loss_avr = loss / log_count
acc_avr = acc / log_count
val_loss_avr = val_loss / log_count
val_acc_avr = val_acc / log_count

print('\n\nloss: {0}, acc: {1}, val_loss: {2}, val_acc: {3}, count{4}\n'.format(loss, acc, val_loss, val_acc, log_count))
print('\nAverage training loss: {}, acc: {}\n'.format(loss_avr, acc_avr))
print('\nAvarage testing val_loss: {}, val_acc: {}\n'.format(val_loss_avr, val_acc_avr))

# save the model to disks
print("[info] serializing network...")
model.save(args["model"])
 
# save the label binarizer to disk
print("[info] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()
