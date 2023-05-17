import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os


def __compile_model(layers, alpha):
    model = models.Sequential()

    for layer in layers:
        model.add(layer)

    opt = Adam(learning_rate=alpha)

    model.build()
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    return model


def __create_callbacks(alpha):
    filepath = f'{os.getcwd()}/output/last_generated_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)
    erl_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1)
    callbacks = [checkpoint, lr_reduce, erl_stopping]
    return callbacks


def __generate_model(alpha, epoch, layers, train_generator, validation_generator):
    model = __compile_model(layers, alpha)
    callbacks = __create_callbacks(alpha)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        callbacks=callbacks,
        epochs=epoch
    )

    return history, model


def train_model_for_dataset(model_config, train_generator, validation_generator):
    history, model = __generate_model(
        model_config['alpha'],
        model_config['epochs'],
        model_config['layers'],
        train_generator,
        validation_generator,
    )

    return model
