# PointCAD:  Class Aware Unsupervised Domain Adaptation of Point Cloud Semantic Segmentation

This repository contains a Pytorch implementation of PointCAD.

## Quick Demo
```bash
python train.py task=segmentation models=segmentation/pvcnn model_name=PVCNN data=segmentation/PowerLineSeg.yaml
```
## Result Visualization
![Figure 7](https://user-images.githubusercontent.com/85683381/121850627-a02e4800-cd1f-11eb-8e07-3dd2972f0e5b.jpg)

## Acknowledgment
This job is based on the framework of [torch-points3d](https://github.com/nicolas-chaulet/torch-points3d).
