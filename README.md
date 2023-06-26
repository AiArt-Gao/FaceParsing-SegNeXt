# FaceParsing via SegNeXt

## Introducion
本项目在 CelebAMask-HQ 数据集上训练了用于人脸解析的SegNeXt模型。在指标表现上远远优于之前常用的 BiSeNetv2 模型。<br>
依据下面的步骤可以方便的使用该模型，获得良好的人脸解析结果。<br>
The project provides SegNeXt for face parsing, trained on the CelebAMask-HQ dataset. <br>
The results are as follows, which are much better than previously widely-used BiSeNetv2.
### IoU of 19 classes
|Model|skin|nose|eye_g|l_eye|r_eye|l_brow|r_brow|l_ear|r_ear|**mouth**|**u_lip**|**l_lip**|**hair**|**hat**|**ear_r**|**neck_l**|**neck**|**cloth**|background|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|**SegNeXt**|93.69|89.29|87.12|82.59|82.55|76.81|76.75|80.86|79.30|87.74|82.41|84.83|91.98|81.80|57.74|22.07|84.88|80.70|93.87|
|BiSeNetv2|92.79|88.40|83.51|34.72|33.13|35.91|25.45|43.11|4.26|83.26|78.30|82.06|90.58|74.23|46.40|0|82.04|71.81|92.15|
### mIou and mAcc of all classes
|Model|**mIou**|**mAcc**|
|-----|-----|-----|
|**SegNeXt**|**79.83**|**86.63**|
|BiSeNetv2|**60.12**|**69.94**|

![image](https://github.com/Beyondzjl/segmentation-CelebAMask-HQ-SegNeXt/assets/84648701/e6941e87-9c4b-488e-a93d-693195cabc89)

## Contributors
- [Beyondzjl](https://github.com/Beyondzjl)
- [fei-aiart](https://github.com/fei-aiart)

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
- Download dataset CelebAMask-HQ from [Google Drive](https://drive.google.com/drive/folders/170q_UvzbzWVDveKd2et2lzaqzTiybKlz?usp=drive_link).<br>
  I have divided the original dataset into following structure.
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
- Prepare OpenMMLab dependences
  ```
  pip install -U openmim
  mim install mmcv-full==1.6.0
  pip install timm
  ```
- Prepare project dependences
  ```
  pip install -r requirements.txt
  ```
### Test
- Get train-weight from [Google Drive](https://drive.google.com/file/d/1rp5D48-1renqNCQ3LkJAYK5__QVFN_IV/view?usp=drive_link).
- Run<br>
  ```
  python tools/test.py ${配置文件} ${检查点文件} [--out ${结果文件}] [--eval ${评估指标}]
  ```
  For example:<br>
  ```
  python /xxx/segmentation-CelebAMask-HQ-SegNeXt/tools/test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/mysegconfig/segnext_CelebAMask_test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/iter_160000.pth --eval mIoU
  ## give you the evalution results
  
  python /xxx/segmentation-CelebAMask-HQ-SegNeXt/tools/test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/mysegconfig/segnext_CelebAMask_test.py /xxx/segmentation-CelebAMask-HQ-SegNeXt/iter_160000.pth --show-dir <results_path> --gpu-id 2
   ## save results to the path
  ```
  
- Tips<br>
  If you want to use your own dataset, you need to write new config giving the proper form and path of your dataset. You can get the example of config from
  mysegconfig.
### Train
- Get pretrain model from [Google Drive](https://drive.google.com/drive/folders/1nrq40tCG4dz1TCPhtPVCacIrYWy9rLBD?usp=drive_link).
- Run with one GPU<br>
  ```
  python tools/train.py ${CONFIG_FILE} [optional parameters]
  ```
## Acknowledgments
The project is based on OpenMMLab. Thanks for the excellent work of [OpenMMLab](https://github.com/open-mmlab/mmsegmentation/tree/main), [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt).
  

