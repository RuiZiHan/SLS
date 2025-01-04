# Straightforward Layer-wise Pruning for More Efficient Visual Adaptation


When fine-tuning pre-trained models for vision tasks using Parameter-Efficient Transfer Learning (PETL), pruning the entire last few layers after fine-tuning significantly improves throughput. In most cases, this results in only a minimal and acceptable drop in performance.

If you want to precisely prune specific redundant layers, follow the steps below:

# Requirements
- Python 3.8+
- PyTorch >= 1.8
- Timm
- Matplotlib
- Pandas
- Scikit-learn


[Paper](http://arxiv.org/abs/2407.14330)

