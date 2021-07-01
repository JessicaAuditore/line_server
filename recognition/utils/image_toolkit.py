"""
*image_toolkit.py
*this file provide some utils functions.
*created by longhaixu
*copyright USTC
*16.11.2020
"""

import copy
import numpy as np
from PIL import Image
import cv2


def pil_img2cv_img(img):
    """
    :param img: a PIL image
    :return: an opencv image
    """
    img = copy.deepcopy(img)
    img = img.convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def cv_img2pil_img(img):
    """
    :param img: an opencv image
    :return: a PIL image
    """
    img = copy.deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def channel_one2three(pil_img):
    """
    :param pil_img: a gray PIL image
    :return: a PIL image with three channels
    """
    image = copy.deepcopy(pil_img)
    image = np.array(image)
    image = np.expand_dims(image, 2)
    image = np.concatenate([image] * 3, -1)
    image = Image.fromarray(image)
    return image


def padding_pilimage_with_ratio(img, ratio):
    """
    paint chinese on an opencv image
    :param im: an opencv image
    :param chinese: string which wanna paint on the image
    :param pos: (x, y) top upper of the string
    :param color: BGR color
    :return: an opencv image
    """
    img = pil_img2cv_img(img)
    H, W, _ = img.shape
    margin = H * ratio - W
    if margin > 0:
        pad = int(margin / 2.)
        img = np.hstack([np.ones([H, pad, 3], dtype=np.uint8) * 255, img,
                         np.ones([H, pad, 3], dtype=np.uint8) * 255])
        # img = np.hstack([img, np.ones([H, margin, 3], dtype=np.uint8) * 255])
    elif margin < 0:
        pad = int(-margin / 2.)
        img = np.vstack([np.ones([pad, W, 3], dtype=np.uint8) * 255, img,
                         np.ones([pad, W, 3], dtype=np.uint8) * 255])
    img = cv_img2pil_img(img)
    return img


def pil_invert(img):
    """
    :param img: a PIL image
    :return: 255 - img
    """
    img = copy.deepcopy(img)
    img = np.asarray(img)
    img = 255 - img
    img = Image.fromarray(img)
    return img


def subimage(image, center, theta, width, height):
    theta *= np.pi / 180
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0], v_y[0], s_x],
                        [v_x[1], v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
