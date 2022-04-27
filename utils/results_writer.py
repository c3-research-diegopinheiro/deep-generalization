import pandas as pd
from utils.consts import results_columns
import os
from datetime import datetime
import shutil
from stat import S_IREAD


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


class ResultsWriter:

    def __init__(self):
        self.results_folder = f'output/results_{datetime.now().isoformat().__str__()}'
        os.mkdir(self.results_folder)
        # shutil.copy('model_configs.py', self.results_folder + '/model_configs.txt')
        # os.chmod(self.results_folder + '/model_configs.txt', S_IREAD)

    def __generate_df_by_csv(self):
        df = pd.read_csv(self.results_folder + '/results.csv')
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df

    def __generate_results_csv(self):
        pd.DataFrame(columns=results_columns).to_csv(self.results_folder + '/results.csv')

    def write_metrics_results(self, model_name, cr, cm):
        if not os.path.isfile(self.results_folder + '/results.csv'):
            self.__generate_results_csv()

        df = self.__generate_df_by_csv()
        split_string = [x.split(' ') for x in cr.split('\n')]
        only_values_list = []
        for row in split_string:
            only_values_list.append(list(filter((lambda x: isfloat(x)), row)))
        values = list(filter((lambda x: len(x) > 0), only_values_list))

        sequence = [[model_name, values[0][0], values[1][0], values[3][0], values[4][0],
                     values[0][1], values[1][1], values[3][1], values[4][1],
                     values[0][2], values[1][2], values[2][0], values[3][2], values[4][2],
                     cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                     ]]

        pd.concat([df, pd.DataFrame(data=sequence, columns=df.columns)], ignore_index=True).to_csv(self.results_folder + '/results.csv')

    def write_model(self, model, model_name):
        model.save(self.results_folder + '/' + model_name + '.hdf5')

