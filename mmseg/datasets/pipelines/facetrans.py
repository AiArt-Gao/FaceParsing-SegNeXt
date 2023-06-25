from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
from ..builder import PIPELINES
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def transform_parsing(pred, center, scale, width, height, input_size):
    if center is not None:
        trans = get_affine_transform(center, scale, 0, input_size, inv=1)
        target_pred = cv2.warpAffine(
            pred,
            trans,
            (int(width), int(height)),  # (int(width), int(height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
    else:
        target_pred = cv2.resize(pred, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)

    return target_pred


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[1]), int(output_size[0])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


@PIPELINES.register_module()
class FaceTrans(object):
    """Resize images & seg to multiple of divisor.

    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self,
                 dataset: str = 'train',
                 crop_size: list = [473, 473],
                 scale_factor: float = 0.25,
                 rotation_factor: int = 30,
                 ignore_label: int = 255,
                 transform=None
                 ):
        self.dataset = dataset
        self.crop_size = np.asarray(crop_size)
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.ignore_label = ignore_label
        self.transform = transform

        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]



    def _box2cs(self, box: list) -> tuple:
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)


    def _xywh2cs(self, x: float, y: float, w: float, h: float) -> tuple:
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale


    def __call__(self, results):
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.
        im = results['img']
        h, w, _ = im.shape
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset in 'train':
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
        trans = get_affine_transform(center, s, r, self.crop_size)
        image = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        results['img'] = image
        # Align segmentation map to multiple of size divisor.
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            # masks = [(gt_seg == v) for v in self.class_values]
            # gt_seg = np.stack(masks, axis=-1).astype('float')

            results[key] = cv2.warpAffine(
                gt_seg,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

        results['img_shape'] = image.shape
        results['pad_shape'] = image.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


