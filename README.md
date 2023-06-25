# segmentation-CelebAMask-HQ-SegNeXt
## Introducion
The project provides SegNeXt trained by CelebAMask-HQ. <br>
The reults are as follows, which are much better than Bisenetv2:
|Model|mIoU|mAcc|
|-----|-----|-----|
|SegNeXt|**79.83**|**86.63**|
|Bisenetv2|60.12|69.94|
## Prerequisites
- Linux
- Python >=3.6
- Anaconda or miniconda
## Quick start
### Preparation
- Clone this repo
  ```
  git clone https://github.com/Beyondzjl/segmentation-CelebAMask-HQ-SegNeXt.git
  cd segmentation-CelebAMask-HQ-SegNeXt
  ```
- Download dataset CelebAMask-HQ from [Google Drive](https://drive.google.com/drive/folders/170q_UvzbzWVDveKd2et2lzaqzTiybKlz?usp=drive_link)<br>
  I have divided the original dataset into following structure ：
  ```
  CelebAMask-HQ
  |
  |-train
  |      |-images
  |      |-labels
  |-test
  |      |-images
  |      |-labels
  |-val
  |      |-images
  |      |-labels
  ```
- Prepare openmmlab dependences
  ```pip install -U openmim
     mim install mmcv-full==1.6.0
     pip install timm
- Prepare project dependences
  ```pip install -r requirements.txt```
### Test
- get train-best-pth from [Google Drive](https://drive.google.com/file/d/1rp5D48-1renqNCQ3LkJAYK5__QVFN_IV/view?usp=drive_link)
- run
  ```python tools/test.py ${配置文件} ${检查点文件} [--out ${结果文件}] [--eval ${评估指标}] ```
  For example
  ```python /xxx/segmentation-CelebAMask-HQ-SegNeXt/tools/test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/mysegconfig/segnext_CelebAMask_test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/iter_160000.pth --eval mIoU ## give you the evalution results
     python /xxx/segmentation-CelebAMask-HQ-SegNeXt/tools/test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/mysegconfig/segnext_CelebAMask_test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/iter_160000.pth --show-dir <results_path>/xxx.png> --gpu-id 2 ## save the results to the path```
- tips
  If you want to use your own dataset, you need to write new config giving the proper form and path of your dataset. You can get the example of config file from
  mysegconfig.
### Train
- get pretrain model from Google Drive.
- run with one GPU
  `python tools/train.py ${CONFIG_FILE} [optional parameters]`
## Acknowledgments
The project is based on openmmlab. Thanks for the excellent work of [openmmlab](https://github.com/open-mmlab/mmsegmentation/tree/main)，[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt)https://github.com/Visual-Attention-Network/SegNeXt.
  

