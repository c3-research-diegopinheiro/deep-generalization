import cv2
import numpy as np
from skimage.util import random_noise
import pandas as pd
import os
from utils.mkdir_dataset import mkdir_dataset


def __write_images(dataset_name, noise_amount, image_path_arr):
    image_path_str = '/'.join(image_path_arr)
    default_image_path = f'dataset/default/{image_path_str}'
    img = cv2.imread(default_image_path)
    print(f'Generating with noise of {noise_amount} for {image_path_str}')
    new_image_path = f'dataset/{dataset_name}/{image_path_arr[1]}/{image_path_arr[2]}'
    if not os.path.exists(new_image_path):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise_img = random_noise(gray_image, mode='salt', amount=noise_amount)
        noise_img = np.array(255 * noise_img, dtype='uint8')
        cv2.imwrite(new_image_path, noise_img)


def generate_dataset(dataset_name, dataset_kind, noise_amount):
    mkdir_dataset(dataset_name)
    print('Copying images to the new dataset')
    df = pd.read_csv('dataset/dataframe.csv')
    [
        __write_images(dataset_name, noise_amount, path_array)
        for path_array in df[['Dataset', 'State', 'Path']].to_numpy() if path_array[0] == dataset_kind
    ]
