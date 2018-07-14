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

- Run the following command:

  ```sh
  cd create-dataset
  python create_hdft.py
  ```


### Train Models

```sh
python train_model.py  <train_set_path> <test_set_path>
```

### Predict

After training the model, you can run the following command:

```sh
python predict.py <your image path>
```
