import glob
import os
import shutil

def images(dataset, type):
    f = os.listdir(path + type)
    return list(map(lambda image: f'dataset/default/{dataset}/{type}/{image}' , f))


path = 'dataset/default/train/'
files = images('train', 'glioma') + images('train', 'meningioma') + images('train', 'pituitary')

for file in files:
    file_name = os.path.basename(file)
    shutil.move(file, 'dataset/default/train/yes/' + file_name)
    print('Moved:', file)


path = 'dataset/default/test/'
files = images('test', 'glioma') + images('test', 'meningioma') + images('test', 'pituitary')

for file in files:
    file_name = os.path.basename(file)
    shutil.move(file, 'dataset/default/test/yes/' + file_name)
    print('Moved:', file)