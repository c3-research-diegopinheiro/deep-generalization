from keras.models import Sequential,load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from dataset_path import dataset_path

def test_accuracy(model_name = 'model_one.hdf5', target_size=(150,150,3)):
    model = load_model(model_name)
    yesDir = dataset_path + '/test/yes'
    noDir = dataset_path + '/test/no'
    yesImagesList = []
    noImagesList = []

    right = 0

    for subdir, dirs, files in os.walk(yesDir):
        for file in files:
            yesImagesList.append(os.path.join(subdir, file))

    for subdir, dirs, files in os.walk(noDir):
        for file in files:
            noImagesList.append(os.path.join(subdir, file))

    for x in yesImagesList:
        img = image.load_img(x, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = img/255
        imageType = (model.predict(img) > 0.5).astype("int32")[0][0]
        if (imageType == 1):
            right += 1

    for x in noImagesList:
        img = image.load_img(x, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = img/255
        imageType = (model.predict(img) > 0.5).astype("int32")[0][0]
        if (imageType == 0):
            right += 1

    print('Acurácia:')
    print((right / (len(noImagesList) + len(yesImagesList))) * 100)
