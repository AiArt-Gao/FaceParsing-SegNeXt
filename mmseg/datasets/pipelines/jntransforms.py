import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES

import albumentations as albu
@PIPELINES.register_module()
class JNTrans(object):
    """Resize images & seg to multiple of divisor.

    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self):
        transform = [
            # albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            albu.RandomCrop(height=320, width=320, always_apply=True),

            albu.GaussNoise(p=0.2),
            albu.Perspective(p=0.5),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        self.transform = albu.Compose(transform)
        self.class_values = [i for i in range(11)]

    def __call__(self, results):
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.
        img = results['img']



        # Align segmentation map to multiple of size divisor.
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            # masks = [(gt_seg == v) for v in self.class_values]
            # gt_seg = np.stack(masks, axis=-1).astype('float')
            out = self.transform(image=img, mask=gt_seg)
            results[key], results['img'] = out['mask'], out['image']

        results['img_shape'] = out['image'].shape
        results['pad_shape'] = out['image'].shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str