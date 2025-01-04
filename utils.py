import torch
import random
import numpy as np
import yaml
import os

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
    model_dir = f'./models/{method}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) 

    torch.save(trainable, f'{model_dir}/{dataset}.pt')

    with open(f'{model_dir}/{dataset}.log', 'w') as f:
        f.write(f'{ep} {acc}')
        

def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('./models/%s/%s.pt'%(method, dataset), weights_only=True)
    model.load_state_dict(st, False)
    return model

def get_config(method, dataset_name):
    with open('./configs/%s/%s.yaml'%(method, dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
