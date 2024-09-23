# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

#from mmseg.datasets import DATASETS, CustomDataset
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import Compose
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ConsepTypeDataset(BaseSegDataset):

    METAINFO = {
        'classes': ['background', 'miscellaneous', 'inflammatory', 'epithelial', 'spindle-shape'],
        'palette': [[50, 50, 50], [255, 255, 0], [255, 0, 255], [0, 0, 255], [0, 255, 255]]
    }

    #
    # METAINFO = {
    #     'classes': ['background', 'inflammatory', 'epithelial', 'stromal'],
    #     'palette': [[0, 0, 0], [255, 0, 255], [0, 0, 255], [0, 255, 255]]
    # }

    '''
    METAINFO = {
        'classes': ['background', 'other', 'inflammatory', 'healthy epithelial',
                    'malignant epithelial', 'fibroblast', 'Muscle',
                    'endothelial'],
        'palette': [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                    [0, 192, 192]]
    }
    '''

    '''
    CLASSES = ('background', 'malignant epithelium', 'normal epithelium',
                'fibroblast', 'inflammatory', 'Muscle',
                'endothelial', 'miscellaneous')


    PALETTE = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
               [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
               [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
               [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
               [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
               [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
               [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
               [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
               [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0],
               [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]]
'''


    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 text_suffix='',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path='', text_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        '''Test Code
        super(ConsepTypeDataset, self).__init__(img_suffix=img_suffix,
                                                seg_map_suffix=seg_map_suffix,
                                                reduce_zero_label=reduce_zero_label,
                                                **kwargs)
        '''

        self.text_suffix = text_suffix

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()


    def get_img_text(self, text_path) -> list:
        text_list = []
        with open(text_path, 'r') as lines:
            # 逐行读取文件内容
            for line in lines:
                words = line.split('\t')
                text_list.append(words)
        return text_list

    def load_img_text(self, img_filename, text) -> list:
        max_length = len(text[0]) -2


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
                data_info['class_text'] = self.class_text_for_predict()
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

    # test for predict text
    def class_text_for_predict(self):
        list = []
        new_classes = self._metainfo.get('classes', None)
        for i in range(len(new_classes)):
            string = 'This is a picture of ' + new_classes[i]
            list.append(string)
        return list





def main():
    A = ConsepTypeDataset(data_prefix=dict(img_path='', seg_map_path='', text_path='data/conseptrain/label/test_text.txt'))
    #A = get_img_text('data/conseptrain/label/test_text.txt')
    print()

if __name__ == '__main__':
    main()


