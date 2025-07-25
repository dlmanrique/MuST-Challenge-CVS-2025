#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import logging

from copy import deepcopy
from .surgical_dataset import SurgicalDataset, SurgicalDatasetChunks
from . import utils as utils
from .build import DATASET_REGISTRY

from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch


logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Cvssages(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "Cvssages"
        self.zero_fill = 5
        self.image_type = "jpg"
        self.cfg = cfg
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        #breakpoint()
        return sec 
        
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """

        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx] # (0,3,451,450)
        video_name = self._video_idx_to_name[video_idx]
        complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type)

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images, complete_name)
        
        found_idx = self._image_paths[video_idx].index(path_complete_name)
        

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]} & {sec_idx} & {sec}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'
        assert int(self._image_paths[video_idx][center_idx].split('/')[-1].split('_')[-1].replace('.'+self.image_type,''))==sec, f'Different {self._image_paths[video_idx][center_idx].split("/")[-1].replace("."+self.image_type,"")} {sec}'

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
            length=self._video_length, 
            online = self.cfg.DATA.ONLINE,
        )

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0
 
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks}

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )
        
        # Preprocess images and boxes
        imgs = self._images_and_boxes_preprocessing_cv2(
            imgs
        )
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('video_',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name
            
        """images_to_show = imgs[0] if isinstance(imgs, list) else imgs  # En caso de slowfast o pathway input
        print(f"Mostrando {len(images_to_show)} frames del clip: {frame_identifier}")
        images_to_show = images_to_show.permute(1, 0, 2, 3)

        for i, img in enumerate(images_to_show):
            # Desnormalizar si es necesario (ImageNet mean/std)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_vis = img * std + mean

            img_vis = TF.to_pil_image(img_vis.clamp(0, 1))  # Convierte tensor a imagen PIL
            img_vis.save(f"frame_{i}.png")"""

        return imgs, all_labels, extra_data, frame_identifier