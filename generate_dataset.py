import cv2
import numpy as np
from skimage.util import random_noise
import pandas as pd
import json

from utils.mkdir_dataset import mkdir_dataset


def __write_images(dataset_name, folders, rows):
    folder_kind = rows[0]
    image_path = 'DATASET/' + '/'.join(rows)
    img = cv2.imread(image_path)

    if folders[folder_kind]['noise']:
        noise_img = random_noise(img, mode='salt', amount=folders[folder_kind]['amount'])
        noise_img = np.array(255 * noise_img, dtype='uint8')
        cv2.imwrite(dataset_name + '/' + '/'.join(rows), noise_img)
    else:
        cv2.imwrite(dataset_name + '/' + '/'.join(rows), img)


def generate_dataset(configs):
    for dataset in configs:
        mkdir_dataset(dataset['name'])
        df = pd.read_csv('dataframe.csv')
        [__write_images(dataset['name'], dataset['folders'], rows) for rows in
         df[['Dataset', 'State', 'Path']].to_numpy()]


dataset_configs = json.load(open('dataset_configs.json', 'r'))
generate_dataset(dataset_configs)
