import traceback
import os
import pandas as pd
from sklearn.model_selection import KFold
from builders import metrics_builder, model_builder
from utils.results_writer import ResultsWriter
from model_configs import model_configs
from temp2 import get_train_generator


def run(data_x, data_y, model_config):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(data_x, data_y):
        rw = ResultsWriter(model_config['name'])
        try:

            for noise_amount in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
                train_generator, validation_generator = get_train_generator(data_x, data_y, noise_amount, train_index)
                trained_model = model_builder.train_model_for_dataset(
                    model_config, train_generator, validation_generator
                )
                rw.write_model(trained_model, f'train_{noise_amount}')

                for noise_amount_testing in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
                    test_generator = get_train_generator(data_x, data_y, noise_amount_testing, test_index)
                    trained_model.evaluate(test_generator)

                    cm = metrics_builder.generate_confusion_matrix(trained_model,
                                                                   test_generator,
                                                                   model_config['batch_size'])
                    cr = metrics_builder.generate_classification_report(trained_model,
                                                                        test_generator,
                                                                        model_config['batch_size'])

                    rw.write_metrics_results(
                        f'train_{noise_amount}_test_{noise_amount_testing}',
                        noise_amount,
                        noise_amount_testing,
                        cr,
                        cm
                    )

        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e.__str__())
            print('Removing results folder for this execution')
            rw.delete_results()


if __name__ == '__main__':
    df = pd.read_csv(f'{os.getcwd()}/dataset/dataframe.csv')
    x = df['images']
    y = df['classes']
    run(x, y, model_configs[0])
