import os.path
import traceback

import pandas as pd
from sklearn.model_selection import KFold

from builders import model_builder
from builders.dataset_builder import generate_dataset
from builders.image_generator_builder import get_train_generator, get_test_generator
from builders.metrics_builder import generate_confusion_matrix, generate_classification_report
from utils.results_writer import ResultsWriter

NOISE_LEVELS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]


def noise_datasets_already_exists():
    noise_levels = [0] + [i / 10 for i in range(1, 10)]
    for noise_level in noise_levels:
        folder_name = f"noise_{noise_level}"
        folder_path = os.path.join('./dataset', folder_name)

        if not os.path.exists(folder_path):
            return False

    return True


class Nid:

    def __init__(self, layers, input_shape, epochs, alpha, batch_size, dataframe_path, model_name="default_name"):
        self.model_config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "alpha": alpha,
            "epochs": epochs,
            "input_shape": input_shape,
            "layers": layers
        }
        self.dataframe_path = dataframe_path

    def __generate_noisy_datasets(self):
        for noise_amount in NOISE_LEVELS:
            dataset_name = f'noise_{noise_amount}'
            generate_dataset(self.dataframe_path, dataset_name, noise_amount)

    def execute(self):
        rw = ResultsWriter(self.model_config['model_name'])

        if not noise_datasets_already_exists():
            self.__generate_noisy_datasets()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        dataframe = pd.read_csv(self.dataframe_path)
        data_x = dataframe['images']
        data_y = dataframe['classes']

        for train_index, test_index in kf.split(data_x, data_y):

            rw.write_execution_folder()

            try:
                for noise_amount in NOISE_LEVELS:

                    train_gen, val_gen = get_train_generator(data_x, data_y, noise_amount, train_index)
                    curr_model = model_builder.train_model_for_dataset(self.model_config, train_gen, val_gen)
                    rw.write_model(curr_model, f'train_{noise_amount}')

                    for noise_amount_testing in NOISE_LEVELS:
                        test_gen = get_test_generator(data_x, data_y, noise_amount_testing, test_index)
                        curr_model.evaluate(test_gen)

                        cm = generate_confusion_matrix(curr_model, test_gen, self.model_config['batch_size'])
                        cr = generate_classification_report(curr_model, test_gen, self.model_config['batch_size'])

                        rw.write_metrics_results(
                            f'train_{noise_amount}_test_{noise_amount_testing}',
                            noise_amount,
                            noise_amount_testing,
                            cr,
                            cm
                        )

            except Exception as e:
                traceback.print_tb(e.__traceback__)
                rw.delete_results()

        rw.generate_mean_csv()
