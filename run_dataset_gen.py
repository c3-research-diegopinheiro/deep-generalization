from builders.dataset_builder import generate_dataset
import pandas as pd
from pathlib import Path

def run(used_dataset):
    yesDir = f'{os.getcwd()}/{used_dataset}/train/yes'
    paths = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.png']:
        paths = [*paths, *[path.parts[-3:] for path in Path(used_dataset).rglob(ext)]]

    df = pd.DataFrame(data=paths, columns=['Dataset', 'State', 'Path'])
    df.to_csv(f'{os.getcwd()}/dataset/dataframe.csv')

    for dataset_kind in ['train', 'test']:
        for noise_amount in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
            dataset_name = f'{dataset_kind}_{noise_amount}'
            generate_dataset(dataset_name, dataset_kind, noise_amount)

if __name__ == '__main__':
    run(f'{os.getcwd()}/dataset/default')