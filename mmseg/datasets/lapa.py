# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LaPaDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """


    CLASSES = ('background', 'skin', 'left eyebrow', 'right eyebrow', 'left eye',
               'right eye', 'nose', 'upper lip', 'inner mouth', 'lower lip', 'hair')

    PALETTE = [[0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
               [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
               [102, 0, 51], [255, 204, 255], [255, 0, 102]]

    def __init__(self, **kwargs):
        super(LaPaDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix,to_label_id,indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result+ 1
            # result = [float(val) for val in result]
            # result = np.array(result)
            result = [float(val) for val in result if isinstance(val, str) and val.replace('.', '', 1).isdigit()]
            result = np.array(result)
            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            print(result.shape)
            print(result.min(), result.max())

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        # results['gt_semantic_seg'] = mmcv.imresize(
        #     results['gt_semantic_seg'], (512,512), interpolation='nearest')
        return results['gt_semantic_seg']