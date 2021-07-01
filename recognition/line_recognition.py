from recognition.model.line_config import Net_Flag
import recognition.model.line_utils
import recognition.model.line_network_future as line_network
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from recognition.utils import utils
import recognition.model.line_chars
from recognition.utils import image_toolkit
import cv2
import numpy as np
import os
import pathlib
import math


class Line_recognition:

    def __init__(self, pth_path='recognition/pth/model.pth'):
        self.net = recognition.model.line_utils.load_net(line_network, Net_Flag.output_node)
        self.net = torch.nn.DataParallel(self.net).cuda()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['net'])
        self.net.to(self.device)
        self.net.eval()

        self.val_transform = transforms.Compose([
            transforms.Resize((Net_Flag.img_H, Net_Flag.img_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def paint_chinese_opencv(self, im, chinese, pos, color):
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

    def one_predict(self, img):
        # img.show()
        img = img.convert('RGB')
        img = image_toolkit.padding_pilimage_with_ratio(img, Net_Flag.img_ratio)
        img = image_toolkit.pil_invert(img)
        img = self.val_transform(img)
        img = torch.unsqueeze(img, 0)

        output = self.net(img)
        output = torch.squeeze(output, 2)
        output = torch.transpose(output, 1, 2)
        indexs_out_lists, scores_list = utils.ctc_decode(output, blank=Net_Flag.num_class - 1, name='G')
        chars = []
        for index in indexs_out_lists[0]:
            chars.append(recognition.model.line_chars.index2char_dict[index])
        predict = ''.join(chars)
        return predict

    def predict(self, path, box_list, output_folder='./output'):
        box_list = sorted(box_list, key=lambda box: box[0][1])
        img = cv2.imread(path)
        path = os.path.join(output_folder, pathlib.Path(path).stem + '_result.txt')
        fp = open(path, 'w', encoding='utf-8')
        for box in iter(box_list):
            center = np.mean(box, axis=0)
            width = (abs(box[0][0] - box[1][0]) ** 2 + abs(box[0][1] - box[1][1]) ** 2) ** 0.5
            height = (abs(box[1][0] - box[2][0]) ** 2 + abs(box[1][1] - box[2][1]) ** 2) ** 0.5
            theta = math.degrees(math.atan(abs(box[0][1] - box[1][1]) / abs(box[0][0] - box[1][0])))
            theta = -theta if box[1][1] < box[0][1] else theta
            image = image_toolkit.subimage(img, center, theta, int(width), int(height) + 10)
            image = cv2.copyMakeBorder(image, 20, 20, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # cv2.imshow('a.png', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            fp.write(self.one_predict(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))) + '\n')
        fp.close()
        return path
