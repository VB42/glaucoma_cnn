import os

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

base_model = VGG16()

width = 224
height = 224
channels = 3
'''
img = load_img('rim-flow-data/train/glaucoma/G-1-L.jpg')
x = img_to_array(img) #numpy array
x = x.reshape((1,) + x.shape) #adds on dimension for keras

print(x.shape)
'''

model = Sequential()
input_shape = channels, width, height if K.image_data_format() == 'channels_first' \
    else width, height, channels
model.add(Conv2D(4, (4, 4), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights('sixth_try.h5')

# model = VGG16(weights='imagenet', include_top=True)
# x = Dense(2, activation='softmax', name='predictions')(model.layers[-2].output)

# Then create the corresponding model
# model = Model(input=model.input, output=x)
# model.summary()

opt = SGD(lr=1, momentum=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

traindatagen = ImageDataGenerator(
    rescale=1. / 255,
    samplewise_std_normalization=True)

testdatagen = ImageDataGenerator(
    rescale=1. / 255,
    samplewise_std_normalization=True)

train_generator = traindatagen.flow_from_directory(
    os.path.join('rim-flow-data', 'train'),  # this is the target directory
    target_size=(width, height),
    batch_size=5,
    color_mode='rgb')

validation_generator = testdatagen.flow_from_directory(
    os.path.join('rim-flow-data', 'validation'),
    target_size=(width, height),
    color_mode='rgb')

model.fit_generator(
    train_generator,
    epochs=40,
    validation_data=validation_generator)

model.save_weights('seventh_try.h5')

# fifth_try - 61.29%
# sixth_try - 67%
