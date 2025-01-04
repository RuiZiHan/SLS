# SLS
Source code of "Straightforward Layer-wise Pruning for More Efficient Visual Adaptation"

[Paper](http://arxiv.org/abs/2407.14330)

![img](https://github.com/RuiZiHan/SLS/blob/main/SLS.png)

TL;DR: When fine-tuning pre-trained models for vision tasks using Parameter-Efficient Transfer Learning (PETL), pruning the entire last few layers after fine-tuning significantly improves throughput. In most cases, this results in only a minimal and acceptable drop in performance.

If you want to precisely prune specific redundant layers, follow the steps below:

## Requirements
- Python 3.8+
- PyTorch >= 1.8
- Timm == 0.9.10
- Matplotlib
- Pandas
- Scikit-learn

## Data Preparation

To download the datasets, please refer to https://github.com/luogen1996/RepAdapter?tab=readme-ov-file#data-preparation (thanks to [luogen1996](https://github.com/luogen1996) and [ZhangYuanhan-AI](https://github.com/ZhangYuanhan-AI) for their efforts). Then move the dataset folders to `<YOUR PATH>/vtab-1k/`

## Usage
### Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/ViT-B_16.npz`

### Tuned a Model Using PETL Method (we use the EuroSAT dataset and the Convpass method as an example)
```sh
python train.py --dataset eurosat --method convpass
# --dataset: Dataset name (e.g., eurosat, cifar100)
# --method: Fine-tuning method (e.g., convpass)
```
### Prune the Parameter-Efficient Model Using SLS
```sh
python prune.py --dataset eurosat --method convpass --alpha 0.3 --plot_tsne False --retraining True
# --alpha: A hyperparameter that controls the pruning degree (default: 0.3)
# --plot_tsne: Enable/disable t-SNE visualization
# --retraining: Whether to retrain the pruned model
```
### Example Results
- EuroSAT Dataset:
  - Original accuracy: 95.87%
  - Pruned accuracy w/o retraining: 55.65%
  - Pruned accuracy w/ retraining: 94.59%
## Citation
If you find this work helpful, please cite the following paper:
```
@inproceedings{han2025straightforward,
  title={Straightforward Layer-wise Pruning for More Efficient Visual Adaptation},
  author={Han, Ruizi and Tang, Jinglei},
  booktitle={European Conference on Computer Vision},
  pages={236--252},
  year={2025},
  organization={Springer}
}
```
## Acknowledgments
Part of the code is borrowed from [Convpass](https://github.com/JieShibo/PETL-ViT/tree/main/convpass) and [timm](https://github.com/rwightman/pytorch-image-models).
