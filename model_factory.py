from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Activation, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def compile_model(layers, alpha):
    model = models.Sequential()

    for layer in layers:
        model.add(layer)

    opt = Adam(learning_rate=alpha)

    model.build()
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    print("Modelo Compilado...")
    return model


def data_augment(dataset_path, batch_size, input_shape):
    image_gen = ImageDataGenerator(rotation_range=40, rescale=1 / 255, horizontal_flip=True, vertical_flip=True)

    print(dataset_path)
    train_images = image_gen.flow_from_directory(dataset_path + '/train', target_size=input_shape[:2],
                                                 batch_size=batch_size, class_mode='binary')
    validation_images = image_gen.flow_from_directory(dataset_path + '/validation',
                                                      target_size=input_shape[:2], batch_size=batch_size,
                                                      class_mode='binary')
    test_images = image_gen.flow_from_directory(dataset_path + '/test', target_size=input_shape[:2],
                                                batch_size=batch_size, class_mode='binary')

    return train_images, validation_images, test_images


def create_callbacks(alpha):
    filepath = "model_one.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)
    erl_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1)

    callbacks = [checkpoint, lr_reduce, erl_stopping]
    return callbacks


def generate_model(dataset_path, input_shape, batch_size, alpha, epoch, layers):
    model = compile_model(layers, alpha)
    callbacks = create_callbacks(alpha)
    train_images, validation_images, test_images = data_augment(dataset_path, batch_size, input_shape)

    print("Iniciando treino do Modelo...")
    history = model.fit(
        train_images,
        validation_data=validation_images,
        callbacks=callbacks,
        epochs=epoch)

    model.evaluate(test_images)

    return history, model
