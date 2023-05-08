from builders.dataset_builder import generate_dataset
import pandas as pd
import os
from pathlib import Path


def run(used_dataset):
    f = []
    for (_, _, filenames) in os.walk(used_dataset):
        f.extend(filenames)

    f = list(map(lambda x: f'{used_dataset}/{x}', f))

    classes = ['no' if 'n' in os.path.basename(file) else 'yes' for file in f]

    df = pd.DataFrame({'images': f, 'classes': classes})
    df.to_csv(f'{os.getcwd()}/dataset/dataframe.csv')

    for noise_amount in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        dataset_name = f'noise_{noise_amount}'
        generate_dataset(dataset_name, noise_amount)


if __name__ == '__main__':
    run(f'{os.getcwd()}/dataset/default')