from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import pandas as pd
# import seaborn as sn
# import matplotlib.pyplot as plt

note = 'Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of ' \
       'the negative class is “specificity”.';


def __mount_test_dataset_generator(test_dataset_path, batch_size):
    image_gen = ImageDataGenerator(rescale=1 / 255)
    return image_gen.flow_from_directory(test_dataset_path,
                                         target_size=(200, 200),
                                         shuffle=False,
                                         class_mode='binary',
                                         batch_size=batch_size,
                                         save_to_dir=None)


def __get_model_predictions(model, test_generator, batch_size):
    predictions = model.predict(test_generator, 300 // batch_size + 1)
    all_predictions = []
    for pred in predictions:
        all_predictions.append((pred[0] > 0.5).astype("int32"))

    return all_predictions


def __get_confusion_matrix(model, test_generator, batch_size):
    predictions = __get_model_predictions(model, test_generator, batch_size)
    return confusion_matrix(test_generator.classes, predictions)


def __get_classification_report(model, test_generator, batch_size):
    predictions = __get_model_predictions(model, test_generator, batch_size)
    target_names = ['Yes', 'No']
    return classification_report(test_generator.classes, predictions, target_names=target_names)


def generate_confusion_matrix(name, model, test_dataset, batch_size):
    if isinstance(test_dataset, str):
        test_generator = __mount_test_dataset_generator(test_dataset, batch_size)
        print('Confusion Matrix')
        cm = __get_confusion_matrix(model, test_generator, batch_size)
        # f = open(name + ".txt", "a")
        # f.write(cm.__str__())
        # f.close()
        print(cm)
        return cm

    else:
        print('Confusion Matrix')
        cm = __get_confusion_matrix(model, test_dataset, batch_size)
        # f = open(name + ".txt", "a")
        # f.write(cm.__str__())
        # f.close()
        print(cm)
        return cm


def generate_classification_report(name, model, test_dataset, batch_size):
    print(note)
    if isinstance(test_dataset, str):
        test_generator = __mount_test_dataset_generator(test_dataset, batch_size)
        print('Classification Report')
        cr = __get_classification_report(model, test_generator, batch_size)
        # f = open(name + ".txt", "a")
        # f.write(cr.__str__())
        # f.close()
        print(cr)
        return cr

    else:
        print('Classification Report')
        cr = __get_classification_report(model, test_dataset, batch_size)
        # f = open(name + ".txt", "a")
        # f.write(cr.__str__())
        # f.close()
        print(cr)
        return cr
