#!/usr/bin/python
#coding:utf-8

import math
import numpy as np
import h5py

def load_dataset(train_set, test_set):
    """
    load train set and test set for the tensorflow session, both of these sets are h5 format.

    Arguments:
    train_set -- train set file path, such as "datasets/train_set.h5"
    test_set -- test set file path, such as "datasets/test_set.h5"

    Returns:
    train_set_x_orig --
    train_set_y_orig --
    test_set_x_orig --
    test_set_y_orig --
    classes --
    """
    print (train_set, test_set)
    train_dataset = h5py.File(train_set, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_set, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
