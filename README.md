# Straightforward Layer-wise Pruning for More Efficient Visual Adaptation


When fine-tuning pre-trained models for vision tasks using Parameter-Efficient Transfer Learning (PETL), pruning the entire last few layers after fine-tuning significantly improves throughput. In most cases, this results in only a minimal and acceptable drop in performance.

If you want to precisely prune specific redundant layers, follow the steps below:

## Requirements
- Python 3.8+
- PyTorch >= 1.8
- Timm == 0.9.10
- Matplotlib
- Pandas
- Scikit-learn

## Data Preparation

To download the datasets, please refer to https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation (thank [ZhangYuanhan-AI](https://github.com/ZhangYuanhan-AI) for their code). Then move the dataset folders to `<YOUR PATH>/vtab-1k/`

## Usage
### Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/ViT-B_16.npz`

### Tuned a Model Using Convpass (we use the EuroSAT dataset as an example)
```sh
python train.py --dataset eurosat --method convpass
```
### Prune the Parameter-Efficient Model Using SLS
```sh
python prune.py --dataset eurosat --method convpass --alpha 0.3 --plot_tsne False --retraining True
```

[Paper](http://arxiv.org/abs/2407.14330)

