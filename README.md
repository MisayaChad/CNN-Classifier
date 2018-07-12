# CNN-Classifier
An image classifier base on tensorflow

# NOTE: this's a project in development!!! So don't launch until I remove this line!

# for some reason I won't upload my datasets, and later I will write a image processer

## CNN Architecture

input layer => conv layer1 => max pool layer1 => conv layer2 => maxpool layer2 => fc(fully connected) layer => softmax => output layer

## Quick Start

### Install packages

```sh
sudo pip install -r requirements.txt
```

### Process datasets

waiting....


### Train Models

```sh
python train_model.py  <train_set_path> <test_set_path>
```

### Predict

After training the model, you can run the following command:

```sh
python predict.py <your image path>
```
