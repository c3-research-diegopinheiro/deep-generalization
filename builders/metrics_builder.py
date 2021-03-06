from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.consts import sensitivity_specificity_note


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


def generate_confusion_matrix(model, test_dataset, batch_size):
    if isinstance(test_dataset, str):
        test_generator = __mount_test_dataset_generator(test_dataset, batch_size)
        print('Generating Confusion Matrix')
        cm = __get_confusion_matrix(model, test_generator, batch_size)
        return cm

    else:
        print('Generating Confusion Matrix')
        cm = __get_confusion_matrix(model, test_dataset, batch_size)
        return cm


def generate_classification_report(model, test_dataset, batch_size):
    print(sensitivity_specificity_note)
    if isinstance(test_dataset, str):
        test_generator = __mount_test_dataset_generator(test_dataset, batch_size)
        print('Generating Classification Report')
        cr = __get_classification_report(model, test_generator, batch_size)
        return cr

    else:
        print('Generating Classification Report')
        cr = __get_classification_report(model, test_dataset, batch_size)
        return cr
