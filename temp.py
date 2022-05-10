from tensorflow.keras.preprocessing.image import ImageDataGenerator

# (input_train, target_train), (input_test, target_test) = cifar10.load_data()
# print(input_train)
# print(target_train)
# input_train = input_train.astype('float32')
#
# input_train = input_train / 255
#
image_gen = ImageDataGenerator(rotation_range=40, rescale=1 / 255, horizontal_flip=True, vertical_flip=True)

train_images = image_gen.flow_from_directory('dataset/default/train', target_size=(200, 200, 3)[:2],
                                                      batch_size=15, class_mode='binary')

print(train_images.next())
print(train_images.classes)
