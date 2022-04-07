from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Activation, Dropout, \
    GlobalAveragePooling2D
import builders.model_builder as model_builder
import builders.metrics_builder as metrics_builder
from generate_dataset import generate_dataset
import json

dataset_configs = json.load(open('dataset_configs.json', 'r'))
generate_dataset(dataset_configs)

for dataset in dataset_configs:
    batch_size = 15
    alpha = 1e-3
    epochs = 30
    input_shape = (200, 200, 3)
    layers = [Flatten(input_shape=input_shape),
              Dense(64, activation='relu'),
              Dense(1, activation='sigmoid')]

    history, model, train_images, validation_images, test_images = model_builder.generate_model(
        'DATASET/' + dataset['name'], input_shape, batch_size, alpha, epochs, layers
    )

    image_gen = ImageDataGenerator(rescale=1 / 255)
    test_generator = image_gen.flow_from_directory('DATASET/' + dataset['name'] + '/test', target_size=(200, 200),
                                                   shuffle=False,
                                                   class_mode='binary',
                                                   batch_size=batch_size,
                                                   save_to_dir=None)
    print('\n')
    metrics_builder.generate_confusion_matrix(dataset['name'], model, test_generator, batch_size)

    print('\n')
    metrics_builder.generate_classification_report(dataset['name'], model, test_generator, batch_size)
