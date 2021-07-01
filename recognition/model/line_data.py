"""
*line_data.py
*this file is for handle data.
*created by longhaixu
*copyright USTC
*16.11.2020
"""

import os
import torch
import recognition.model.line_chars
from PIL import Image
from recognition.model.line_config import Net_Flag, Training_Flag
from torchvision import transforms

import sys

sys.path.append("..")

from ..utils import image_toolkit
import recognition.model.line_utils


class GetLoader(torch.utils.data.Dataset):
    """
    A data loader provided for get batch data which is required while training model.
    """

    def __init__(self, root, folders_list, is_train):
        """
        :param root: The path of datasets.
        :param folders_list: a list of path,
                example:[r'Gnt1.0Test', r'Gnt1.0TrainPart1', r'Gnt1.0TrainPart2', r'Gnt1.0TrainPart3',
                                 r'Gnt1.1Test', r'Gnt1.1TrainPart1', r'Gnt1.1TrainPart2',
                                 r'Gnt1.2Test', r'Gnt1.2TrainPart1', r'Gnt1.2TrainPart2']
        :param is_train: True or False
        """
        self.is_train = is_train
        self.imgs_list, self.labels_list, self.lens_list = get_imgs_labels_list(root, folders_list)
        print('number of total image: %d' % len(self.imgs_list))

    def __getitem__(self, index):
        """
        :param index: index of data in dataset
        :return: data for training
        """
        image = Image.open(self.imgs_list[index])
        image = image_toolkit.padding_pilimage_with_ratio(image, Net_Flag.img_ratio)
        image = image_toolkit.pil_invert(image)
        if self.is_train:
            image = Transform.train_transform(image)
        else:
            image = Transform.val_transform(image)
        return image, self.labels_list[index], self.lens_list[index]

    def __len__(self):
        return len(self.imgs_list)


class Transform(object):
    """
    A transform for each image. There are two difference kind of transforms for training and testing.
    """
    train_transform = transforms.Compose([
        transforms.ColorJitter(contrast=0.3, brightness=0.2, saturation=0.3),
        transforms.RandomAffine(degrees=5),
        transforms.RandomPerspective(distortion_scale=0.2, interpolation=3, p=0.2),
        transforms.Resize((Net_Flag.img_H, Net_Flag.img_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Net_Flag.img_H, Net_Flag.img_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_imgs_labels_list(root, folder_list):
    '''
    :param root: path to search, it's the parent folder of etc.'Gnt1.0Test'.
                example:r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/hwdb1'
    :param folder_list: you can just select the folders such as 'Gnt1.0Test' to search.
                example:[r'Gnt1.0Test', r'Gnt1.0TrainPart1', r'Gnt1.0TrainPart2', r'Gnt1.0TrainPart3',
                         r'Gnt1.1Test', r'Gnt1.1TrainPart1', r'Gnt1.1TrainPart2',
                         r'Gnt1.2Test', r'Gnt1.2TrainPart1', r'Gnt1.2TrainPart2']
    This function will generate a txt file of chars' dicts.
    '''
    imgs_list = []
    labels_list = []
    lens_list = []

    def search(root):
        paths = os.listdir(root)
        for path in paths:
            if path.endswith('jpg'):
                imgs_list.append(os.path.join(root, path))
                label = []
                for x in list(path.split('_')[2]):
                    if x == 'ÿ' or x in ['伿', '怳', '時', '呎', '幹', '遡']:
                        continue
                    x = line_utils.change_font(x)
                    label.append(line_chars.char2index_dict[x])
                lens_list.append(len(label))
                label += [Net_Flag.end_flag] * (Net_Flag.seq_length - len(label))
                labels_list.append(label)
            else:
                if os.path.isdir(os.path.join(root, path)):
                    search(os.path.join(root, path))

    roots = [os.path.join(root, x) for x in folder_list]
    for r in roots:
        search(r)
    return imgs_list, labels_list, lens_list


def generate_chars_dict(root, folder_list):
    '''
    :param root: path to search, it's the parent folder of etc.'Gnt1.0Test'.
                example:r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/hwdb1'
    :param folder_list: you can just select the folders such as 'Gnt1.0Test' to search.
                example:[r'Gnt1.0Test', r'Gnt1.0TrainPart1', r'Gnt1.0TrainPart2', r'Gnt1.0TrainPart3',
                         r'Gnt1.1Test', r'Gnt1.1TrainPart1', r'Gnt1.1TrainPart2',
                         r'Gnt1.2Test', r'Gnt1.2TrainPart1', r'Gnt1.2TrainPart2']
    This function will generate a txt file of chars' dicts.
    '''
    char_dict = {}
    chars = []
    global count

    def search(root):
        global count
        paths = os.listdir(root)
        for path in paths:
            if path.endswith('jpg'):
                print(count)
                if path.startswith('_'):
                    count += 1
                c = path.split('_')[1]
                if c not in chars:
                    char_dict[len(chars)] = path.split('_')[1]
                    chars.append(c)
            else:
                if path.startswith('Gnt') and path not in folder_list:
                    continue
                search(os.path.join(root, path))

    search(root)
    print('total samples: %d, length of dictionaries %d' % (count, len(char_dict)))
    with open('./tmp.txt', 'a') as f:
        f.write(str(char_dict))


def reverse_dict(raw_dict):
    """
    :param raw_dict: a dictionary
    :return: a dictionary which keys define as raw_dict's values and
             values define as raw_dict's keys
    """
    new_dict = {}
    for k, v in raw_dict.items():
        new_dict[v] = k
    with open('./tmp.txt', 'a') as f:
        f.write(str(new_dict))

# def generate_line_data(root, train_floder_list):
#     imgs_list, labels_list = get_imgs_labels_list(root, train_floder_list)
#     for i in range(10):
#         indexs = np.random.randint(0, len(imgs_list), 10, np.int)
#         print(indexs)
#         for ind in indexs:
#             print(imgs_list[ind], labels_list[ind])
