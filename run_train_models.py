import traceback
import os
import pandas as pd
from sklearn.model_selection import KFold
from builders import model_builder
from builders.metrics_builder import generate_confusion_matrix, generate_classification_report
from utils.results_writer import ResultsWriter
from model_configs import model_configs
from builders.image_generator_builder import get_train_generator, get_test_generator

NOISE_LEVELS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]


def run(data_x, data_y, model_config):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(data_x, data_y):

        rw = ResultsWriter(model_config['name'])

        try:
            for noise_amount in NOISE_LEVELS:

                train_gen, val_gen = get_train_generator(data_x, data_y, noise_amount, train_index)
                curr_model = model_builder.train_model_for_dataset(model_config, train_gen, val_gen)
                rw.write_model(curr_model, f'train_{noise_amount}')

                for noise_amount_testing in NOISE_LEVELS:

                    test_gen = get_test_generator(data_x, data_y, noise_amount_testing, test_index)
                    curr_model.evaluate(test_gen)

                    cm = generate_confusion_matrix(curr_model, test_gen, model_config['batch_size'])
                    cr = generate_classification_report(curr_model, test_gen, model_config['batch_size'])

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


if __name__ == '__main__':
    df = pd.read_csv(f'{os.getcwd()}/dataset/dataframe.csv')
    x = df['images']
    y = df['classes']
    run(x, y, model_configs[0])
