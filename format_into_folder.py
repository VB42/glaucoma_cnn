# Run only once to sort out data
# Glauc Test Data images = ['G-39-L.jpg', 'S-29-R.jpg', 'G-33-R.jpg', 'S-16-R.jpg', 'G-24-L.jpg', 'S-25-L.jpg',
#                           'S-31-L.jpg', 'S-5-L.jpg', 'S-18-L.jpg', 'G-13-R.jpg', 'G-18-R.jpg', 'G-35-R.jpg',
#                           'G-7-L.jpg', 'S-6-R.jpg']
# Health Test Data images = ['N-47-L.jpg', 'N-70-R.jpg', 'N-58-R.jpg', 'N-11-L.jpg', 'N-22-R.jpg', 'N-69-L.jpg',
#                            'N-13-L.jpg', 'N-81-L.jpg', 'N-6-R.jpg', 'N-21-L.jpg', 'N-18-R.jpg', 'N-44-R.jpg',
#                            'N-3-L.jpg', 'N-90-R.jpg', 'N-23-L.jpg', 'N-55-L.jpg', 'N-32-R.jpg']
import os
import random

from PIL import Image
from dotenv import load_dotenv

load_dotenv('.config.env')
glauc_path = os.getenv('GLAUC_PATH')
health_path = os.getenv('HEALTH_PATH')

dst = os.getenv('FORMATTED_DATA_PATH')

glauc_dir = os.listdir(glauc_path)
health_dir = os.listdir(health_path)

glauc_train = random.sample(glauc_dir, len(glauc_dir))
health_train = random.sample(health_dir, len(health_dir))

glauc_test = random.sample(glauc_dir, len(glauc_dir) // 5)
health_test = random.sample(health_dir, len(health_dir) // 5)

glauc_train = list(set(glauc_train).difference(set(glauc_test)))
health_train = list(set(health_train).difference(set(health_test)))


def crop(image_path, loc):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop((0, 0, image_obj.size[0] / 2, image_obj.size[1]))
    cropped_image.save(loc)


for x in glauc_train:
    crop(os.path.join(glauc_path, x), os.path.join(dst, 'train', 'glaucoma', x))
    # copy(glauc_path + '/' + x, dst + '/train/glaucoma/')

for x in glauc_test:
    crop(os.path.join(glauc_path, x), os.path.join(dst, 'validation', 'glaucoma', x))
    # copy(glauc_path + '/' + x, dst + '/validation/glaucoma')

for x in health_train:
    crop(os.path.join(health_path, x), os.path.join(dst, 'train', 'healthy', x))
    # copy(health_path + '/' + x, dst + '/train/healthy')

for x in health_test:
    crop(os.path.join(health_path, x), os.path.join(dst, 'validation', 'healthy', x))
    # copy(health_path + '/' + x, dst + '/validation/healthy')
