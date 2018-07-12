#!/usr/bin/python
#coding:utf-8

from utils import *

hdf5_path = './dataset.h5'
images_path = '/Users/alexliu/Desktop/*.jpg'
keyword = 'cat'

file_sets = list_images_and_lables(images_path, keyword)
create_hdf5(hdf5_path, file_sets)
load_images_info_h5(hdf5_path, file_sets)
