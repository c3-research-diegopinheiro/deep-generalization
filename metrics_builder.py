from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def __get_confusion_matrix(model, test_generator, batch_size):
    predictions = model.predict(test_generator, 300 // batch_size + 1)
    x = []
    for pred in predictions:
        x.append((pred[0] > 0.5).astype("int32"))

    return confusion_matrix(test_generator.classes, x)


def __get_classification_report(model, test_generator, batch_size):
    predictions = model.predict(test_generator, 300 // batch_size + 1)
    x = []
    for pred in predictions:
        x.append((pred[0] > 0.5).astype("int32"))

    target_names = ['Yes', 'No']
    classification_report(test_generator.classes, x, target_names=target_names)


def generate_confusion_matrix(model, test_dataset, batch_size):
    if isinstance(test_dataset, str):
        image_gen = ImageDataGenerator(rescale=1 / 255)
        test_generator = image_gen.flow_from_directory(test_dataset, target_size=(200, 200),
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       batch_size=batch_size,
                                                       save_to_dir=None)
        print('Confusion Matrix')
        print(__get_confusion_matrix(model, test_generator, batch_size))

    else:
        print('Confusion Matrix')
        print(__get_confusion_matrix(model, test_dataset, batch_size))


def generate_classification_report(model, test_dataset, batch_size):
    if isinstance(test_dataset, str):
        image_gen = ImageDataGenerator(rescale=1 / 255)
        test_generator = image_gen.flow_from_directory(test_dataset, target_size=(200, 200),
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       batch_size=batch_size,
                                                       save_to_dir=None)
        print('Classification Report')
        print(__get_classification_report(model, test_generator, batch_size))

    else:
        print('Classification Report')
        print(__get_classification_report(model, test_dataset, batch_size))


