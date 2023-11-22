import torch
import random
import numpy as np
import yaml
from pathlib import Path


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def save(method, dataset, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable[n] = p.data
    # /root/autodl-tmp/output/conv_output
    model_path = Path('/root/autodl-tmp/output/conv_output/%s/%s.pt' % (method, dataset))
    log_path = Path('/root/autodl-tmp/output/conv_output/%s/%s.log' % (method, dataset))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainable, model_path)
    with open(log_path, 'w') as f:
        f.write(str(ep) + ' ' + str(acc))


def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('/root/autodl-tmp/output/conv_output/%s/%s.pt' % (method, dataset))
    model.load_state_dict(st, False)
    return model


def get_config(method, dataset_name):
    with open('./configs/%s/%s.yaml' % (method, dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
