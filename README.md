# segmentation-CelebAMask-HQ-SegNeXt
## Introducion
## Dataset
## Prerequisites
- Linux
- Python >=3.6
- Anaconda or miniconda
## Quick start
### Preparation
- Clone this repo
  ' git clone https://github.com/Beyondzjl/segmentation-CelebAMask-HQ-SegNeXt.git'
  'cd segmentation-CelebAMask-HQ-SegNeXt'
- Download dataset CelebAMask-HQ
  I have divided the original dataset into following structure ：
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
  mmsegmentation 
│
├─ data 
│   └─ my_dataset                 # 转换后的自己的数据集文件 
├─ mmseg 
│   └─ datasets 
│       ├─ __init__.py            # 在这里加入自己的数据集的类 
│       ├─ my_dataset.py          # 定义自己的数据集的类 
│       └─ ... 
├─ configs 
│   ├─ _base_ 
│   │   └─ datasets 
│   │       └─ my_dataset_config.py    # 自己的数据集的配置文件 
│   └─ ... 
└─ ...

### Test
get train-best-pth from [Google Drive](https://drive.google.com/file/d/1rp5D48-1renqNCQ3LkJAYK5__QVFN_IV/view?usp=drive_link)
### Train

