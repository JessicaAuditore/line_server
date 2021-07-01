"""
*line_pipeline.py
*this file is a pipeline file.
*created by longhaixu
*copyright USTC
*16.11.2020
"""

import os
from line_config import Net_Flag
import line_utils
# import line_network
import line_network_future as line_network
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from ..utils import utils
import line_chars
from ..utils import image_toolkit
import cv2
import numpy as np


def paint_chinese_opencv(im, chinese, pos, color):
    """
    paint chinese on an opencv image
    :param im: an opencv image
    :param chinese: string which wanna paint on the image
    :param pos: (x, y) top upper of the string
    :param color: BGR color
    :return: an opencv image
    """
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 30)
    fillColor = color
    position = pos
    # if not isinstance(chinese, unicode):
    chinese = chinese.encode('GBK').decode('GBK')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

pth_path = r'/home/hxlong/Project/NEW_HWR/line_recognition/save/save_real_3conv.pth'

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

net = line_utils.load_net(line_network, Net_Flag.output_node)
net = torch.nn.DataParallel(net).cuda()
checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])
net.eval()

val_transform = transforms.Compose([
    transforms.Resize((Net_Flag.img_H, Net_Flag.img_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# test_image_root = r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/test/line/test'
test_image_root = r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/line/HWDB2.0Test_pure'
# test_image_root = r'./src'
# test_image_root = r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/line/Generate_01_corpus'
imgs_path = [os.path.join(test_image_root, x) for x in os.listdir(test_image_root)]
for path in imgs_path:
    img = Image.open(path)
    img = img.convert('RGB')
    img = image_toolkit.padding_pilimage_with_ratio(img, Net_Flag.img_ratio)
    img = image_toolkit.pil_invert(img)
    img = val_transform(img)
    print(img.shape)
    img = torch.unsqueeze(img, 0)

    output = net(img)
    output = torch.squeeze(output, 2)
    output = torch.transpose(output, 1, 2)
    print(output.shape)
    # output = torch.reshape(output, (-1, Net_Flag.seq_length, Net_Flag.num_class))
    indexs_out_lists, scores_list = utils.ctc_decode(output, blank=Net_Flag.num_class-1, name='G')

    # indexs_out_lists_2, _ = utils.ctc_decode(output, blank=Net_Flag.num_class - 1, name='B')
    # chars = []
    # for index in indexs_out_lists[0]:
    #     chars.append(line_chars.index2char_dict[index])
    # beam_out = ''.join(chars)
    # print(beam_out)

    chars = []
    for index in indexs_out_lists[0]:
        chars.append(line_chars.index2char_dict[index])
    print(path)
    label = list(path.split('/')[-1].split('_')[2])
    for i, char in enumerate(label):
        label[i] = line_utils.change_font(char)
    label = ''.join(label)

    predict = ''.join(chars)
    AR, CR = utils.cal_AR_CR_for_two_str(predict, label)
    # print(utils.cal_AR_CR_for_two_str(beam_out, label))
    # print('label  : {}\npredict: {}\nAR is {:.4f}, CR is {:.4f}'.format(label, predict, AR, CR))
    print('label  : {}\npredict: {}\nAR is {:.4f}, CR is {:.4f}'.format(label, predict, AR, CR))
    # input()

    img = Image.open(path)
    img = img.convert('RGB')
    img = image_toolkit.padding_pilimage_with_ratio(img, Net_Flag.img_ratio)
    cv_img = image_toolkit.pil_img2cv_img(img)
    cv_img = cv2.resize(cv_img, (1300, 100))
    label_img = paint_chinese_opencv(np.vstack([cv_img, np.ones_like(cv_img, dtype=np.uint8) * 255]),
                                     '预测结果:  '+label, (20, 120), (0, 0, 0))
    cv2.imshow('Predict', label_img)
    cv2.waitKey(6000)



