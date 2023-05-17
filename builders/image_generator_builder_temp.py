from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

NOISE_LEVELS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]


def get_train_generator(x, y, train_index):
    x_trains = []
    y_trains = []
    for n in NOISE_LEVELS:
        img = list(map(lambda p: p.replace('default', f'noise_{n}'), x))
        df = pd.DataFrame({'images': img})
        x = df['images']
        x_train, y_train = x[train_index], y[train_index]
        x_trains.extend(x_train)
        y_trains.extend(y_train)

    df_train = pd.DataFrame({'id': x_trains, 'label': y_trains})

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

    return train_generator, validation_generator


def get_test_generator(x, y, noise, test_index):
    x = list(map(lambda p: p.replace('default', f'noise_{noise}'), x))
    df = pd.DataFrame({'images': x})
    x = df['images']

    x_test, y_test = x[test_index], y[test_index]

    df_test = pd.DataFrame({'id': x_test, 'label': y_test})

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

    return test_generator
