import pandas as pd
from utils.consts import results_columns
import os


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def __generate_df_by_csv():
    df = pd.read_csv('output/results.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def __generate_results_csv():
    pd.DataFrame(columns=results_columns).to_csv('output/results.csv')


def write_metrics_results(model_name, cr, cm):
    if not os.path.isfile('output/results.csv'):
        __generate_results_csv()

    df = __generate_df_by_csv()
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

    pd.concat([df, pd.DataFrame(data=sequence, columns=df.columns)], ignore_index=True).to_csv('output/results.csv')


