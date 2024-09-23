import warnings
import torch
import numpy as np
from typing import Any, Union, List

from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData
import mmcv
from mmseg.registry import TRANSFORMS



@TRANSFORMS.register_module()
class MyRandomFlip(BaseTransform):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def transform(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            for i in range(0, len(results['text_main'])):
                if 'right' in results['text_main'][i] or 'left' in results['text_main'][i]:
                    text = results['text_main'][i]
                    text = text.replace('right', 'temp')
                    text = text.replace('left', 'right')
                    text = text.replace('temp', 'left')
                    results['text_main'][i] = text

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
        return results

