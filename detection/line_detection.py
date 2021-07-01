import os
import cv2
import torch
import pathlib

from .utils.util import get_transforms, resize_image, get_post_processing, draw_bbox
from .model import build_model


class Line_detection:

    def __init__(self, model_path='detection/pth/model.pth'):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = 0.3
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, path, is_output_polygon=False, short_size=1024, output_folder='./output'):
        img = cv2.imread(path, 1 if self.img_mode != 'GRAY' else 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, _ = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list = box_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
                    box_list = box_list[idx]
            else:
                box_list, score_list = [], []

        img = draw_bbox(cv2.imread(path)[:, :, ::-1], box_list)
        output_path = os.path.join(output_folder, pathlib.Path(path).stem + '_result.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])

        return output_path, box_list
