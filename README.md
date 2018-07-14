# CNN-Classifier
An image classifier base on tensorflow. You can finish any type of image classifier base on it. The only thing you need to do is collecting your image data.

## CNN Architecture

input layer => conv layer1 => max pool layer1 => conv layer2 => maxpool layer2 => fc(fully connected) layer => softmax => output layer

## Quick Start

### Install packages

```sh
sudo pip install -r requirements.txt
```

### Process datasets

- Put your images into create-dataset dir and give them suit file names. for example, if your image shows a dog, then the file name should contain "dog"

here list all my files in `create-dateset` directory, assuming we have three class; dog, cat, and others.

  ```sh
  ├── __init__.py
  ├── create_hdf5.py
  ├── images
  │   ├── test_cat1.jpg
  │   ├── test_cat2.jpg
  │   ├── test_cat6.jpg
  │   ├── test_dog.jpg
  │   └── test_hahah.jpg
  ├── list_images_and_lable.py
  ├── load_images.py
  ├── utils.py
  └── utils.pyc
  ```

- Modify `create_hdf5.py`

```python
#!/usr/bin/python
#coding:utf-8

from utils import *

train_hdf5_path = './train_dataset.h5'
test_hdf5_path = './test_dataset.h5'

images_path = './*.jpg'

file_sets = list_images_and_lables(images_path, keyword)
create_hdf5(train_hdf5_path, test_hdf5_path, file_sets)
load_images_info_h5(train_hdf5_path, test_hdf5_path, file_sets)
```

modify `train_hdf5_path` and `test_hdf5_path` parameters whatever you like, and don't forget put your image into `image_path`

- Run the following command:

  ```sh
  cd create-dataset
  python create_hdft.py
  ```


### Train Models

Before you train the model, you can adjust the `minibatch_size` parameter, set it to 1 if you have few images and just want to test it.

there are some guidelines:

1. if you have less than (<=) 2000 samples, set this parameter to 1, which means you just use batch gradient descent.

2. choose a number from 64 to 512, use 2^n, such as 64, 128, 256, 512 which can speed up your training

```sh
python train_model.py  <train_set_path> <test_set_path>
```

### Predict

After training the model, you can run the following command:

```sh
python predict.py <your image path>
```

### Which parameters should you choose to improve your performance?

there are somes tips:

1. learning rate
2. mini-batch size
3. hidden layers

## Just have fun :)
