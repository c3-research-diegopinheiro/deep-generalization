from keras.preprocessing.image import  ImageDataGenerator, img_to_array, image, load_img
import pandas as pd


def get_dataset_generators(x, y, noise, train_index, test_index):
    x = list(map(lambda p: p.replace('default', f'noise_{noise}'), x))
    df = pd.DataFrame({'images': x })
    x = df['images']

    x_train, y_train, x_test, y_test = \
        x[train_index], y[train_index], x[test_index], y[test_index]

    df_train = pd.DataFrame({'id': x_train, 'label': y_train})
    df_test = pd.DataFrame({'id': x_test, 'label': y_test})

    train_gen = ImageDataGenerator(
        rotation_range=40,
        rescale=1 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1
    )

    train_generator = train_gen.flow_from_dataframe(
        dataframe=df_train,
        x_col='id',
        y_col='label',
        batch_size=30,
        target_size=(200, 200),
        class_mode='binary',
        shuffle=False,
        subset='training')

    validation_generator = train_gen.flow_from_dataframe(
        dataframe=df_train,
        x_col='id',
        y_col='label',
        batch_size=30,
        target_size=(200, 200),
        class_mode='binary',
        shuffle=False,
        subset='validation')

    test_generator = ImageDataGenerator(
        rescale=1 / 255,
    ).flow_from_dataframe(
        dataframe=df_test,
        x_col='id',
        y_col='label',
        batch_size=30,
        target_size=(200, 200),
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator, test_generator

