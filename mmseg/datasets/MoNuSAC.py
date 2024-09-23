from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import Compose

@DATASETS.register_module()
class MoNuSACDataset(BaseSegDataset):
    """Conic2022SegDataset dataset.

    In segmentation map annotation for Conic2022SegDataset,
    ``reduce_zero_label`` is fixed to False. The ``img_suffix``
    is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png'.

    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
    """
    METAINFO = dict(
        classes=('background', 'Epithelial', 'Lymphocyte', 'Neutrophil',
                 'Macrophage', 'Ambiguous'),
        # palette= [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
        #         [0, 192, 192]]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 # data_prefix: dict = dict(img_path='', seg_map_path=''),
                 # data_prefix: dict = dict(img_path='', seg_map_path='', text_path=''),
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            # data_prefix = copy.copy(data_prefix),
            **kwargs)

        # self.predict_text = self.class_text_for_predict()
'''
    def get_img_text(self, text_path) -> list:
        text_list = []
        with open(text_path, 'r') as lines:
            # 逐行读取文件内容
            for line in lines:
                words = line.split('\t')
                text_list.append(words)
        return text_list

    def load_img_text(self, img_filename, text) -> list:
        max_length = 25


        for line in text:
            if line[0] == img_filename:
                img_text = line[1:]
                while len(img_text) <= max_length:
                    img_text.append("")
                if len(img_text) > max_length + 1:
                    img_text = img_text[:max_length]
                return img_text
        return []

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        text_path = self.data_prefix.get('text_path', None)
        text_list = self.get_img_text(text_path)

        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_info['text_main'] = self.load_img_text(img, text_list)
                data_info['class_text'] = self.predict_text
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

    # test for predict text
    def class_text_for_predict(self):
        list = []
        new_classes = self._metainfo.get('classes', None)
        for i in range(len(new_classes)):
            string = 'This is a picture of ' + self.new_classes[i]
            list.append(string)
        return list
        
'''