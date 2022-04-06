import os


def mkdir_dataset(new_dataset_folder):
    if not os.path.exists(new_dataset_folder):
        dataset_structure = {
            'train': ['yes', 'no'],
            'test': ['yes', 'no'],
            'validation': ['yes', 'no'],
        }
        for key in dataset_structure:
            for classification in dataset_structure[key]:
                os.makedirs(new_dataset_folder + '/' + key + '/' + classification)
    else:
        print('dataset folder already exists')
