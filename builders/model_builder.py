from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model


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


def __data_augment(train_path, validation_path, batch_size, input_shape):
    image_gen = ImageDataGenerator(rotation_range=40, rescale=1 / 255, horizontal_flip=True, vertical_flip=True)

    train_images = image_gen.flow_from_directory(train_path, target_size=input_shape[:2],
                                                 batch_size=batch_size, class_mode='binary')
    validation_images = image_gen.flow_from_directory(validation_path,
                                                      target_size=input_shape[:2], batch_size=batch_size,
                                                      class_mode='binary')

    return train_images, validation_images


def __create_callbacks(alpha):
    filepath = 'output/last_generated_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)
    erl_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1)
    callbacks = [checkpoint, lr_reduce, erl_stopping]
    return callbacks


def __generate_model(input_shape, batch_size, alpha, epoch, layers, train_path, validation_path):
    model = __compile_model(layers, alpha)
    callbacks = __create_callbacks(alpha)
    train_images, validation_images = __data_augment(train_path, validation_path, batch_size, input_shape)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    print("Iniciando treino do Modelo...")
    history = model.fit(
        train_images,
        validation_data=validation_images,
        callbacks=callbacks,
        epochs=epoch
    )

    return history, model, train_images, validation_images


def train_model_for_dataset(model_config, train_folder_path, validation_folder_path):
    print(f'Training for {train_folder_path} and {validation_folder_path} dataset')
    history, model, train_images, validation_images = __generate_model(
        model_config['input_shape'],
        model_config['batch_size'],
        model_config['alpha'],
        model_config['epochs'],
        model_config['layers'],
        train_folder_path,
        validation_folder_path,
    )

    return model
