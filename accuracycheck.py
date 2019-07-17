from __future__ import print_function

import os

from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

width = 356 * 4
height = 536 * 4
channels = 1

input_shape = channels, width, height if K.image_data_format() == 'channels_first' \
    else width, height, channels

'''
img = load_img('rim-flow-data/train/glaucoma/G-1-L.jpg')
x = img_to_array(img) #numpy array
x = x.reshape((1,) + x.shape) #adds on dimension for keras

print(x.shape)'''

model = Sequential()
model.add(Conv2D(4, (10, 10), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights('third_try.h5')

opt = optimizers.SGD(lr=0.01, momentum=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

traindatagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    samplewise_std_normalization=True)

testdatagen = ImageDataGenerator(
    rescale=1. / 255,
    samplewise_std_normalization=True)

train_generator = traindatagen.flow_from_directory(
    os.path.join('rim-flow-data', 'train'),  # this is the target directory
    target_size=(width, height),
    batch_size=10,
    color_mode='grayscale')

validation_generator = testdatagen.flow_from_directory(
    os.path.join('rim-flow-data', 'validation'),
    target_size=(width, height),
    batch_size=10,
    color_mode='grayscale')

model.fit_generator(
    train_generator,
    epochs=1)


def format_image(path):
    img = load_img(path, target_size=(width, height), grayscale=True)
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    return img


print(train_generator.class_indices)
print(validation_generator.class_indices)

print('TRAIN\n______________________')

print('Glauc')

print(model.predict(format_image(os.path.join('rim-flow-data', 'train', 'glaucoma', 'G-1-L.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'train', 'glaucoma', 'G-2-R.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'train', 'glaucoma', 'G-3-R.jpg'))))

print()

print('Health')
print(model.predict(format_image(os.path.join('rim-flow-data', 'train', 'healthy', 'N-1-L.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'train', 'healthy', 'N-2-R.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'train', 'healthy', 'N-4-R.jpg'))))

print()
print()

print('VALIDATION\n______________________')

print('Glauc')
print(model.predict(format_image(os.path.join('rim-flow-data', 'validation', 'glaucoma', 'G-7-L.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'validation', 'glaucoma', 'G-13-R.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'validation', 'glaucoma', 'G-18-R.jpg'))))

print()

print('Health')
print(model.predict(format_image(os.path.join('rim-flow-data', 'validation', 'healthy', 'N-3-L.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'validation', 'healthy', 'N-6-R.jpg'))))
print(model.predict(format_image(os.path.join('rim-flow-data', 'validation', 'healthy', 'N-11-L.jpg'))))
