#coding:utf-8

# Extract cifar10 data from original python version to jpgs.

import os
import pickle
import numpy as np
import cv2


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def makeDataDirs(base_path, train_dir, test_dir, category_count):

    train_path = os.path.join(base_path, train_dir)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(base_path, test_dir)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for i in range(category_count):
        category_path_train = os.path.join(train_path, str(i))
        if not os.path.exists(category_path_train):
            os.makedirs(category_path_train)

        category_path_test = os.path.join(test_path, str(i))
        if not os.path.exists(category_path_test):
            os.makedirs(category_path_test)


def main(read_path, save_path):

    # label
    file_dict = unpickle(os.path.join(read_path, "batches.meta"))
    #print(file_dict) 
    with open(os.path.join(save_path,"label_names.txt"), "w", encoding="utf-8") as f:
        for i,label_name in enumerate(file_dict[b"label_names"]):
            f.write(str(i)+":"+str(label_name, encoding = "utf-8")+"\n")


    # img data
    file_names = ['data_batch_1',
                'data_batch_2',
                'data_batch_3',
                'data_batch_4',
                'data_batch_5',
                'test_batch',]

    train_dir = "train"
    test_dir = "test"
    category_count = len(file_dict[b"label_names"])
    makeDataDirs(save_path, train_dir, test_dir, category_count)

    for file_name in file_names:
        file_path = os.path.join(read_path, file_name)
        file_dict = unpickle(file_path)
        #print(file_dict.keys()) 
        #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

        for i in range(len(file_dict[b'labels'])):
            img_label = file_dict[b'labels'][i]
            img_data = file_dict[b'data'][i]
            img_name = str(file_dict[b'filenames'][i], encoding = "utf-8")

            img_rgb = np.reshape(img_data, (3,32,32)).transpose(1,2,0)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if "test" not in file_name:
                img_save_path = os.path.join(save_path, train_dir, str(img_label), img_name)
            else:
                img_save_path = os.path.join(save_path, test_dir, str(img_label), img_name)
            cv2.imwrite(img_save_path, img_bgr)
            #b


if __name__ == '__main__':

    read_path = r"F:\Data\cifar10\cifar-10-batches-py"
    save_path = r"F:\Data\cifar10\cifar10_imgs"

    main(read_path, save_path)