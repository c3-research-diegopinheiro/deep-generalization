import glob
import os
import shutil

def images(dataset, type):
    f = os.listdir(path + type)
    return list(map(lambda image: f'{os.getcwd()}/dataset/default/{dataset}/{type}/{image}' , f))


path = f'{os.getcwd()}/dataset/default/train/'
files = images('train', 'glioma') + images('train', 'meningioma') + images('train', 'pituitary')

for file in files:
    file_name = os.path.basename(file)
    shutil.move(file, os.getcwd() + '/dataset/default/train/yes/' + file_name)
    print('Moved:', file)


path = f'{os.getcwd()}/dataset/default/test/'
files = images('test', 'glioma') + images('test', 'meningioma') + images('test', 'pituitary')

for file in files:
    file_name = os.path.basename(file)
    shutil.move(file, os.getcwd() + '/dataset/default/test/yes/' + file_name)
    print('Moved:', file)