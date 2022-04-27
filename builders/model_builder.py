from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


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

    print("Modelo Compilado")
    return model


def __data_augment(folder_name, batch_size, input_shape):
    image_gen = ImageDataGenerator(rotation_range=40, rescale=1 / 255, horizontal_flip=True, vertical_flip=True)

    train_images = image_gen.flow_from_directory(f'DATASET/{folder_name}/train', target_size=input_shape[:2],
                                                 batch_size=batch_size, class_mode='binary')
    validation_images = image_gen.flow_from_directory(f'DATASET/{folder_name}/validation',
                                                      target_size=input_shape[:2], batch_size=batch_size,
                                                      class_mode='binary')

    test_images = ImageDataGenerator(rescale=1 / 255).flow_from_directory(f'DATASET/{folder_name}/test',
                                                                          target_size=(200, 200),
                                                                          shuffle=False,
                                                                          class_mode='binary',
                                                                          batch_size=batch_size,
                                                                          save_to_dir=None)

    return train_images, validation_images, test_images


def __create_callbacks(alpha):
    filepath = 'output/last_generated_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)
    erl_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1)

    callbacks = [checkpoint, lr_reduce, erl_stopping]
    return callbacks


def generate_model(input_shape, batch_size, alpha, epoch, layers, folder_name):
    model = __compile_model(layers, alpha)
    callbacks = __create_callbacks(alpha)
    train_images, validation_images, test_images = __data_augment(folder_name, batch_size, input_shape)

    print("Iniciando treino do Modelo...")
    history = model.fit(
        train_images,
        validation_data=validation_images,
        callbacks=callbacks,
        epochs=epoch)

    model.evaluate(test_images)

    return history, model, train_images, validation_images, test_images
