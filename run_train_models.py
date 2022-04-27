from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from builders import metrics_builder, model_builder
from utils.results_writer import ResultsWriter
from pathlib import Path

rw = ResultsWriter()

network = {
    "batch_size": 15,
    "alpha": 1e-3,
    "epochs": 30,
    "input_shape": (200, 200, 3),
    "layers": [
        Flatten(input_shape=(200, 200, 3)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
}


def train_model_for_dataset(folder_name):
    dataset_folder_name = folder_name.split('/')[1]
    print(f'Training for {dataset_folder_name} dataset')
    history, model, train_images, validation_images, test_images = model_builder.generate_model(
        network['input_shape'],
        network['batch_size'],
        network['alpha'],
        network['epochs'],
        network['layers'],
        dataset_folder_name,
    )

    cm = metrics_builder.generate_confusion_matrix(model, test_images, network['batch_size'])
    cr = metrics_builder.generate_classification_report(model, test_images, network['batch_size'])

    rw.write_metrics_results(dataset_folder_name, cr, cm)
    rw.write_model(model, dataset_folder_name)


p = Path('./DATASET')
[train_model_for_dataset(str(f)) for f in p.iterdir() if f.is_dir()]
# train_model_for_dataset(str(list(p.iterdir())[2]))
