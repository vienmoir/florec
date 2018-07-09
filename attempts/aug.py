# coding: utf-8
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from cropim import CropIm
from imutils import paths
import cv2
import random
import os
import numpy as np

datagen = ImageDataGenerator(rotation_range = 90, width_shift_range = 0.1,
                       height_shift_range = 0.1, shear_range = 0.2, 
                        zoom_range = 0.2, horizontal_flip = True, 
                         fill_mode = "nearest")

# impath = '..\\testim\\sc.jpg'
#  # this is a Numpy array with shape (3, 150, 150)
# x = CropIm(impath)
# #x = im.reshape(im.shape[1], im.shape[2], im.shape[0]) 
# print(x.shape)
# x = x.reshape((1,) + x.shape)
# print(x.shape) # this is a Numpy array with shape (1, 3, 150, 150)

# #img = load_img('..\\testim\\sc.jpg')  # this is a PIL image
# #y = img_to_array(img)
# # print(y.shape)  # this is a Numpy array with shape (3, 150, 150)
# # y = y.reshape((1,) + y.shape) 
# # print(y.shape) # this is a Numpy array with shape (1, 3, 150, 150)
args = {
    "dataset": "D:\\Uni\\TU\\DL\\project\\florec\\testim",
    "model": "D:\\Uni\\TU\\DL\\project\\florec\\florec.model",
    "labelbin": "D:\\Uni\\TU\\DL\\project\\florec\\lb.pickle",
    "plot": "D:\\Uni\\TU\\DL\\project\\florec\\plot.png"
}
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


#x = datagen.flow(data, labels, batch_size = 3)

# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(data, labels, batch_size=1,
                          save_to_dir='..\\testimres', save_prefix='test', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely