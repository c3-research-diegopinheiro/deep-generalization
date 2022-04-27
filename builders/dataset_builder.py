import cv2
import numpy as np
from skimage.util import random_noise
import pandas as pd
import os
from utils.mkdir_dataset import mkdir_dataset


def __write_images(dataset_name, folders, image_path_arr):
    folder_kind = image_path_arr[0]
    default_image_path = 'DATASET/default/' + '/'.join(image_path_arr)
    img = cv2.imread(default_image_path)

    new_image_path = 'DATASET/' + dataset_name + '/' + '/'.join(image_path_arr)
    if not os.path.exists(new_image_path):
        if folders[folder_kind]['noise']:
            noise_img = random_noise(img, mode='salt', amount=folders[folder_kind]['amount'])
            noise_img = np.array(255 * noise_img, dtype='uint8')
            cv2.imwrite(new_image_path, noise_img)
        else:
            cv2.imwrite(new_image_path, img)


def generate_dataset(model_config):
    mkdir_dataset(model_config['name'])
    print('Copying images to the new dataset')
    df = pd.read_csv('dataframe.csv')
    [__write_images(model_config['name'], model_config['dataset_structure'], rows) for rows in df[['Dataset', 'State', 'Path']].to_numpy()]

