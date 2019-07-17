# Glaucoma CNN

This Github repository has the work for a Convolutional Neural Network to detect the presence of glaucoma in fundus images. I created the CNN for a school science fair project that I worked with two peers.

## Datasets
We looked for numerous datasets that classified images as healthy or with glaucoma, and settled on the dataset known as RIM-ONE, provided by the Medical Image Analysis Group of the University of La Laguna (http://medimrg.webs.ull.es/)

## Data Sorting
The format_into_folder.py file was used for formatting the data into a way that Keras could use the flow_from_directory option. This code was only run once for the sorting of data. Additionally, the data given had two images attached next to each other from slightly different angles, so I cropped the right half of the image off.

## CNN
The network was built with Keras, using TensorFlow as a backend. The model achieved up to 77% accuracy during a training session, but the most accuracy the model finished with was around 67%. The different weights are given, but refer to older models, so use the latest weights for loading the model with the current architecture. Note, the model didn't achieve that high of an accuracy probably do to a lack of data and/or due to not enough layers/less training time. After some time, I shifted the model running from main.py to mainv2.py, so mainv2.py should be the latest running version.

## Visualization
The visualization was done in the file cnnvisualizer.py. It generated saliency maps, CAMs, an activation maps on different layers. This was built using the keras-vis package, built by Kotikalapudi, Raghavendra, and contributors found at https://github.com/raghakot/keras-vis.


#### Example CAM Images:

**Glaucoma:**
![Glaucoma Image](layer-genimages/layer_2/cam/glaucoma/G-1-L.jpg)

**Healthy:**
![Healthy Image](/layer-genimages/layer_2/cam/healthy/N-1-L.jpg)


*Note that these images were from the 2nd layer and are respectively the G-1-L.jpg image and the N-1-L.jpg image from the training phase.*

## Install
Ensure you a version of Python in the 2.7-3.6 range installed, then run:

    pip install -r requirements.txt
