from abc import abstractmethod, ABC
import numpy as np
import torch
import pickle as pkl
from gdl.utils.FaceDetector import FaceDetector, MTCNN
import os, sys
from gdl.utils.other import get_path_to_externals 
from pathlib import Path
from torchvision import transforms as tf
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from face_alignment.utils import get_preds_fromhm, crop
from collections import OrderedDict
import torch.nn.functional as F
from munch import Munch

path_to_sbr = (Path(get_path_to_externals()) / ".." / ".." / "landmark-detection" / "SBR").absolute()
# path_to_hrnet = (Path(get_path_to_externals())  / "landmark-detection"/ "SBR").absolute()

if str(path_to_sbr) not in sys.path:
    sys.path.insert(0, str(path_to_sbr))

from lib.models   import obtain_model, remove_module_dict
# from lib.config_utils import obtain_lk_args as obtain_args

INPUT_SIZE = 256


class SBR(FaceDetector):

    def __init__(self, device = 'cuda', instantiate_detector='sfd', threshold=0.5):


        snapshot =  path_to_sbr / "snapshots/300W-CPM-DET/checkpoint/cpm_vgg16-epoch-049-050.pth"

        snapshot = torch.load(snapshot)

        self.net = obtain_model(model_config, lk_config, args.num_pts + 1)

        try:
            weights = remove_module_dict(snapshot['detector'])
        except:
            weights = remove_module_dict(snapshot['state_dict'])
        self.net.load_state_dict(weights)


        self.detector = None
        if instantiate_detector == 'mtcnn':
            self.detector = MTCNN()
        elif instantiate_detector == 'sfd': 
            # Get the face detector

            face_detector_kwargs =  {
                "filter_threshold": threshold
            }
            self.detector = SFDDetector(device=device, verbose=False, **face_detector_kwargs)

        elif instantiate_detector is not None: 
            raise ValueError("Invalid value for instantiate_detector: {}".format(instantiate_detector))
        

        self.transforms = [tf.ToTensor()]

        self.crop_to_tensor = tf.Compose(self.transforms)


    @torch.no_grad()
    def run(self, image, with_landmarks=False, detected_faces=None):

        if detected_faces is None: 
            bboxes = self.detector.detect_from_image(image)
        else:
            print("Image size: {}".format(image.shape)) 
            bboxes = [np.array([0, 0, image.shape[1], image.shape[0]])]

        final_boxes = []
        final_kpts = []

        for bbox in bboxes:
            center = torch.tensor(
                [bbox[2] - (bbox[2] - bbox[0]) / 2.0, bbox[3] - (bbox[3] - bbox[1]) / 2.0])

            center[1] = center[1] + (bbox[3] - bbox[1])  * 0.00 # this appears to be a good value

            scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / self.detector.reference_scale * 0.65 # this appears to be a good value

            images_ = crop(image, center, scale, resolution=256.0)
            images = self.crop_to_tensor(images_)
            if images.ndimension() == 3:
                images = images.unsqueeze(0)

            pts_img, X_lm_hm = self.detect_in_crop(images, center.unsqueeze(0), torch.tensor([scale]))

            if pts_img is None:
                del pts_img
                if with_landmarks:
                    return [],  f'kpt{self.num_landmarks}', []
                else:
                    return [],  f'kpt{self.num_landmarks}'
            else:
                import matplotlib.pyplot as plt

                for i in range(len(pts_img)):
                    kpt = pts_img[i][:68].squeeze().detach().cpu().numpy()
                    left = np.min(kpt[:, 0])
                    right = np.max(kpt[:, 0])
                    top = np.min(kpt[:, 1])
                    bottom = np.max(kpt[:, 1])
                    final_bbox = [left, top, right, bottom]
                    final_boxes += [final_bbox]
                    final_kpts += [kpt]



        if with_landmarks:
            return final_boxes, f'kpt{self.num_landmarks}', final_kpts
        else:
            return final_boxes, f'kpt{self.num_landmarks}'


    @torch.no_grad()
    def detect_in_crop(self, crop, center, scale):
        with torch.no_grad():
            output = self.model(crop)
            
            batch_heatmaps, batch_locs, batch_scos = self.net(crop)


        return preds, score_map

