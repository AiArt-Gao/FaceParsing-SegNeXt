# Copyright (c) OpenMMLab. All rights reserved.
# aiart_h
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    imgs = sorted(os.listdir(args.img))
    if not os.path.exists(args.out_file):
        os.mkdir(args.out_file)
    for img in imgs:
        path = os.path.join(args.img,img)
        result = inference_segmentor(model, path)
        out = np.expand_dims(result[0], 2).repeat(3, axis=2)
        out = out.astype(np.uint8)
        plt.imsave(os.path.join(args.out_file, img[:-4]+'.png'), out)
    # show the results

    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     get_palette(args.palette),
    #     opacity=args.opacity,
    #     out_file=args.out_file)


if __name__ == '__main__':
    main()
