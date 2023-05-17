import os


def mkdir_dataset(new_dataset_folder_name):
    new_dataset_folder_path = f'{os.getcwd()}/dataset/{new_dataset_folder_name}'
    if not os.path.exists(new_dataset_folder_path):
        os.mkdir(new_dataset_folder_path)
        print('dataset folder created inside dataset: ' + new_dataset_folder_name)
    else:
        print('dataset folder already exists: ' + new_dataset_folder_name)


def mkdir_output():
    try:
        os.mkdir(f'{os.getcwd()}/output')
    except FileExistsError:
        pass
