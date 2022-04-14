# DeepWindows

Dataset and PyTorch Implementation of [DeepWindows: Windows Instance Segmentation through an Improved Mask R-CNN
 Using Spatial Attention and Relation Modules](https://www.mdpi.com/2220-9964/11/3/162).
 
 
 ## Contents
- [Dataset](#Dataset)
- [Citation](#Citation) 
- [Requirements](#Requirements)  
- [Usage](#Usage) 

 
 ![](https://www.mdpi.com/ijgi/ijgi-11-00162/article_deploy/html/images/ijgi-11-00162-g001-550.jpg)
## Dataset
The dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1bVJa5kM2FvpZM75RQoJ7Oq9m1PTs240-/view?usp=sharing)
 or [百度网盘](https://pan.baidu.com/s/1nCyMIlvacimHjVGvpIzymw) (提取码：srfo)

You need to change the dataset path in train_net.py to your own path.

## Citation

If you use our code or dataset, please use the following BibTeX entry.

```BibTeX
@Article{sun2022deepwindows,
  author =       {Sun, Yanwei and Malihi, Shirin and Li, Hao and Maboudi, Mehdi},
  title =        {DeepWindows: Windows Instance Segmentation through an Improved Mask R-CNN Using Spatial Attention and Relation Modules},
  journal =      {ISPRS International Journal of Geo-Information},
  volume =       {11},
  year =         {2022},
  number =       {3},
  article-number = {162},
  url =          {https://www.mdpi.com/2220-9964/11/3/162}
  issn =         {2220-9964},
  doi =          {10.3390/ijgi11030162}
}
``` 

## Requirements
- Linux(tested on Ubuntu 20.04)
- Python 3.7
- PyTorch
- [detectron2](https://github.com/facebookresearch/detectron2)
- OpenCV (for visualization)

## Usage

### Train
to train Mask R-CNN:
```bash
./train_net.py \
  --config-file ./configs/mask_rcnn_R_50_FPN_1x.yaml \
```
to train deepwindows network:
```bash
./train_net.py \
  --config-file ./configs/CASARPN_RM_R_50_FPN_1x.yaml \
```
### Evaluation
to calculate average precision:
```bash
./train_net.py \
  --config-file ./configs/CASARPN_RM_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
to calculate pixel accuracy:
```bash
./calcPixelAccuracy.py \
--input /JSON file produced by the model
--dataset /name of the dataset
```

### Predict
```bash
./predict_results.py \
  --config-file ./configs/CASARPN_RM_R_50_FPN_1x.yaml
  --input /path/to/input/images
  --output /path/to/output
  --opts
  MODEL.WEIGHTS /path/to/checkpoint_file
```
