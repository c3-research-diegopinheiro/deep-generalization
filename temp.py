import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import sklearn.model_selection as sklrn
from model_configs import model_configs
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import models

train_dir = 'dataset/default/train'
TARGET_SIZE=(200,200)

imgs = [img_fname for img_fname in os.listdir(f'{train_dir}')]
label = ['yes' if 'no_' not in img_fname else 'no' for img_fname in os.listdir(f'{train_dir}')]

train_X = np.array(imgs)
train_labels = np.array(label)


def train_and_cross_validate(model, x_data, y_data, n_folds=5, epochs=15, batch_size=30):
    #
    scores = []

    #  Loading images through generators ...
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # prepare cross validation
    kfold = sklrn.KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    FoldsSetNo = 0
    for train_ix, test_ix in kfold.split(x_data):
        print('Folds Set # {0}'.format(FoldsSetNo))
        # select rows for train and test
        xx_train, yy_train, xx_test, yy_test = \
            x_data[train_ix], y_data[train_ix], x_data[test_ix], y_data[test_ix]

        print(xx_train)
        # flow training images in batches for the current folds set
        # for training
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'id': xx_train, 'label': yy_train}),
            directory=train_dir,
            x_col='id',
            y_col='label',
            batch_size=batch_size,
            target_size=TARGET_SIZE,
            class_mode='binary',
            shuffle=False)

        # and for validation
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'id': xx_test, 'label': yy_test}),
            directory=train_dir,
            x_col='id',
            y_col='label',
            batch_size=batch_size,
            target_size=TARGET_SIZE,
            class_mode='binary',
            shuffle=False)

        # fit the model
        history = model.fit(train_generator,
                            epochs=epochs,  # The more we train the more our model fits the data
                            batch_size=batch_size,  # Smaller batch sizes = samller steps towards convergence
                            validation_data=validation_generator,
                            verbose=1)
        # store scores
        scores.append(
            {'acc': np.average(history.history['accuracy']), 'val_acc': np.average(history.history['val_accuracy'])})
        FoldsSetNo += 1
    return scores


used_model = model_configs[0]
m = models.Sequential(used_model['layers'])

opt = Adam(learning_rate=used_model['alpha'])

m.build()
m.summary()
m.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['acc'])

print("Modelo Compilado")

train_and_cross_validate(m, train_X, train_labels)
