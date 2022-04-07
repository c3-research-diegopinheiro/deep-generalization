import os


def mkdir_dataset(new_dataset_folder_name):
    new_dataset_folder_path = 'DATASET/' + new_dataset_folder_name
    if not os.path.exists(new_dataset_folder_path):
        dataset_structure = {
            'train': ['yes', 'no'],
            'test': ['yes', 'no'],
            'validation': ['yes', 'no'],
        }
        for key in dataset_structure:
            for classification in dataset_structure[key]:
                os.makedirs(new_dataset_folder_path + '/' + key + '/' + classification)
    else:
        print('dataset folder already exists: ' + new_dataset_folder_name)
