import os
from utils.consts import dataset_structure


def mkdir_dataset(new_dataset_folder_name):
    new_dataset_folder_path = 'DATASET/' + new_dataset_folder_name
    if not os.path.exists(new_dataset_folder_path):
        for key in dataset_structure:
            for classification in dataset_structure[key]:
                os.makedirs(new_dataset_folder_path + '/' + key + '/' + classification)
        print('dataset folder created inside DATASET/: ' + new_dataset_folder_name)
    else:
        print('dataset folder already exists: ' + new_dataset_folder_name)
