import os

from keras import activations
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, visualize_activation, visualize_cam

# import pydot
#
# from keras import backend as K
#
# import tensorflow as tf
#
#
# from keras.callbacks import TensorBoard
#
#
# from guided_backprop import GuidedBackprop
# from utils import *
#
# import PIL
#
# from dotenv import load_dotenv


width = 224
height = 224

model = Sequential()
model.add(Conv2D(64, (4, 4), input_shape=(width, height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.load_weights('eighth_try.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


def save_img(path, savepath, origimg, typeimg, layeridx):
    img = load_img(path, target_size=(224, 224))
    x = img_to_array(img)  # numpy array
    x = x.reshape(x.shape)  # adds on dimension for keras

    model.layers[layeridx].activation = activations.linear
    if typeimg == 'activation':
        img = visualize_activation(model, layeridx, 20, x)

    if typeimg == 'saliency':
        img = visualize_saliency(model, layeridx, 1, x)

    if typeimg == 'cam':
        img = visualize_cam(model, layeridx, 1, x)

    if not os.path.exists('layer-' + savepath):
        os.makedirs('layer-' + savepath)

    if not os.path.exists('image-' + savepath):
        os.makedirs('image-' + savepath)

    combined = os.path.join(savepath, origimg)
    plt.imshow(img)
    plt.savefig('layer-' + combined, dpi=600)
    # plt.imshow(x)
    # plt.savefig('image-' + combined)


types = ['saliency', 'cam']
glauc_imgs = ['G-1-L.jpg', 'G-2-R.jpg']
health_imgs = ['N-1-L.jpg', 'N-2-R.jpg']
for layeridx in [2, 3, 4, 5, 6]:
    print('LAYER:', layeridx)
    for typeimg in types:
        print('TYPE:', typeimg)
        for imgidx in range(0, 2):
            save_img(os.path.join('rim-flow-datav2', 'train', 'glaucoma', glauc_imgs[imgidx]),
                     os.path.join('genimages' + 'layer_' + str(layeridx), typeimg, 'glaucoma'),
                     glauc_imgs[imgidx],
                     typeimg,
                     layeridx)
            save_img(os.path.join('rim-flow-datav2', 'train', 'healthy', health_imgs[imgidx]),
                     os.path.join('genimages' + 'layer_' + str(layeridx), typeimg, 'healthy'),
                     health_imgs[imgidx],
                     typeimg,
                     layeridx)

input_img = load_img(os.path.join('rim-flow-data', 'train', 'glaucoma', 'G-1-L.jpg'))
input_img = img_to_array(input_img)  # numpy array
input_img = input_img.reshape((1,) + input_img.shape)  # adds on dimension for keras

#
# layer_dict = dict([(layer.name, layer) for layer in model.layers])
#
# layer_name = 'conv2d_1'
# filter_index = 1
#
# layer_output = layer_dict[layer_name].output
# loss = K.mean(layer_output[filter_index, :, :, :])
#
#
# # compute the gradient of the input picture wrt this loss
# grads = K.gradients(loss, input_img)[0]
#
# print('HIIIIII\n\n\\n\n\n\n\\n\n\n\n')
# print(loss)
#
# # normalization trick: we normalize the gradient
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
# # this function returns the loss and grads given the input picture
# iterate = K.function([input_img], [loss, grads])
#
# input_img_data = np.random.random((1, 3, width, height)) * 20 + 128.
# # run gradient ascent for 20 steps
# for i in range(20):
#     loss_value, grads_value = iterate([input_img_data])
#     input_img_data += grads_value * step
#
# from scipy.misc import imsave
#
# # util function to convert a tensor into a valid image
# def deprocess_image(x):
#     # normalize tensor: center on 0., ensure std is 0.1
#     x -= x.mean()
#     x /= (x.std() + 1e-5)
#     x *= 0.1
#
#     # clip to [0, 1]
#     x += 0.5
#     x = np.clip(x, 0, 1)
#
#     # convert to RGB array
#     x *= 255
#     x = x.transpose((1, 2, 0))
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
#
# img = input_img_data[0]
# img = deprocess_image(img)
# imsave('%s_filter_%d.png' % (layer_name, filter_index), img)


#
# load_dotenv('.config.env')
# dst = os.getenv('FORMATTED_DATA_PATH')
#
# guided_bprop = GuidedBackprop(model)
#
# # Load the image and compute the guided gradient

#
# layer_idx = -1
#
# # Swap softmax with linear
# model.layers[layer_idx].activation = activations.linear
# model = utils.apply_modifications(model)
#
# plt.rcParams['figure.figsize'] = (18, 6)
#
# img = visualize_activation(model, layer_idx)
#
# print(img[...,0])
# plt.show(img)
