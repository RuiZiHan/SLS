from argparse import ArgumentParser

import pandas as pd
import torch
from PIL import Image
from einops import rearrange

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch_cka import CKA

from timm import create_model
from torch_cka.utils import add_colorbar
from torchvision.transforms import transforms, InterpolationMode
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from vtab import get_data
from convpass import set_Convpass
from pruning_train import load
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import torchvision.datasets as datasets
from utils import get_config, set_seed
import torch.nn as nn
from sklearn import manifold

from repadapter import set_RepAdapter, set_RepWeight


def visual(feat):
    # t-SNE
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


def plotlabels(S_lowDWeights, Trure_labels, method, model_struct, dataset, layer_id):
    print("start plot t_SNE result")
    config = get_config(method=method, dataset_name=dataset)
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    print(S_data)

    S_data.to_csv(
        f'/root/autodl-tmp/output_ours/visualization_output/t-SNE/{method}/{model_struct}/{dataset}_{layer_id}th_layer.csv')
    print(S_data.shape)  # [num, 3]

    for index in range(config['class_num']):  #class_num
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        # plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)
        plt.scatter(X, Y, cmap='brg', s=100, alpha=0.65)

        plt.xticks([])  
        plt.yticks([])  
        plt.axis('off')

    # name = f'{method}_{model_struct}_{dataset}_{layer_id}blocks'
    # plt.title(name, fontsize=64, fontweight='normal', pad=20)


def forward_ViT_features(self, x):
    # timm==0.4.12
    # x = self.patch_embed(x)
    # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    # if self.dist_token is None:
    #     x = torch.cat((cls_token, x), dim=1)
    # else:
    #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    # x = self.pos_drop(x + self.pos_embed)
    # feature_list = []
    # for i in range(12):
    #     x = self.blocks[i](x)
    #     feature_list.append(x)
    # x = self.norm(x)
    # if self.dist_token is None:
    #     return self.pre_logits(x[:, 0]), feature_list
    # else:
    #     return x[:, 0], x[:, 1]
    # timm==0.9.10
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    feature_list = []
    for i in range(12):
        x = self.blocks[i](x)
        feature_list.append(x)

    x = self.norm(x)
    return x, feature_list


def forward_Swin_features(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    feature_list = []
    for layer in range(4):
        depth = 2 if layer != 2 else 18
        for index in range(depth):
            x = self.layers[layer].blocks[index](x)
            # feature_list.append(torch.flatten(self.avgpool(x.transpose(1, 2)), 1))
            if self.layers[layer].downsample is not None:
                feature_list.append(torch.flatten(self.avgpool(self.layers[layer].downsample(x).transpose(1, 2)), 1))
            else:
                feature_list.append(torch.flatten(self.avgpool(x.transpose(1, 2)), 1))
        if self.layers[layer].downsample is not None:
            x = self.layers[layer].downsample(x)

    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x, feature_list


@torch.no_grad()
def get_feature(method, model_struct, dataset, layer_id, model_depth=12):
    train_dl, val_dl = get_data(dataset, evaluate=True, batch_size=40)

    config = get_config(method=method, dataset_name=dataset)
    if model_struct == 'ViT':
        model = create_model('vit_base_patch16_224_in21k', checkpoint_path='/root/autodl-tmp/pre_train/ViT-B_16.npz',
                             drop_path_rate=0.1)
    elif args.model_struct == 'Swin':
        model = create_model('swin_base_patch4_window7_224_in22k', drop_path_rate=0.1)
        check_point = torch.load('/root/autodl-tmp/pre_train/swin_base_patch4_window7_224_22k.pth')
        model.load_state_dict(check_point['model'])

    if method == 'convpass':
        set_Convpass(model, method, dim=8, s=config['scale'], xavier_init=config['xavier_init'])
    elif method == 'repadapter':
        set_RepAdapter(model, 'repblock', dim=8, s=config['scale'], set_forward=False)
    else:
        raise NotImplementedError
    model.reset_classifier(config['class_num'])
    model = load(dataset=dataset, model=model, method=method, model_struct=model_struct, blocks=model_depth)
    if method == 'repadapter':
        set_RepWeight(model, 'repblock', dim=8, s=config['scale'])
    model.to('cuda')
    model.eval()

    feature_collection = []
    label_collection = []
    layer_id = layer_id
    for batch in tqdm(train_dl):
        input, label = batch[0].to('cuda'), batch[1].to('cuda')
        if model_struct == 'ViT':
            pred, feature_list = forward_ViT_features(model, input)
            feature_collection.append(feature_list[layer_id - 1][:, 0])
            label_collection.append(label)
        elif args.model_struct == 'Swin':
            pred, feature_list = forward_Swin_features(model, input)
            feature_collection.append(feature_list[layer_id - 1])
            label_collection.append(label)
    intermediate_feature = torch.cat(feature_collection, dim=0)
    label = torch.cat(label_collection, dim=0)
    return intermediate_feature.cpu(), label.cpu()


def run_t_SNE(method, model_struct, dataset, layer_id, model_depth=12):
    print('start run t_SNE')
    feat, label_test = get_feature(method=method, model_struct=model_struct, dataset=dataset, layer_id=layer_id,
                                   model_depth=model_depth)
    fig = plt.figure(figsize=(20, 10))
    plotlabels(visual(feat), label_test, method=method, model_struct=model_struct, dataset=dataset, layer_id=layer_id)
    plt.show()
    plt.savefig(f'/root/autodl-tmp/output_ours/visualization_output/t-SNE/{layer_id}.png', dpi=300)


def analyse_t_SNE(method, model_struct, dataset, layer_id, best_score):
    print(f'start analyse {layer_id}th_layer t_SNE')
    # config = get_config(method=method, dataset_name=dataset)
    # cls_num = config['class_num']
    df = pd.read_csv(
        f'/root/autodl-tmp/output_ours/visualization_output/t-SNE/{method}/{model_struct}/{dataset}_{layer_id}th_layer.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    # center = df.groupby('label').mean().to_numpy()
    X = df[['x', 'y']].to_numpy()
    label = df['label'].to_numpy()
    # SC index
    scoreSC = silhouette_score(X, label)
    # CH index
    # scoreCH = calinski_harabasz_score(X, label)
    # DB index
    # scoreDB = davies_bouldin_score(X, label)

    print(f"{layer_id}th layer")
    print(f"SC Index[-1,1]: {scoreSC}")
    # print(f"CH Index[0,+] by sklearn: {scoreCH}")
    # print(f"DB Index[0,+] by sklearn: {scoreDB}")
    break_alalyse = False
    if model_struct == 'ViT':
        threshold = 0.3 * best_score
    elif model_struct == 'Swin':
        threshold = -1
    if scoreSC >= threshold:
        print(f"current score:{scoreSC}>=threshold:{threshold}")
        with open(f'/root/autodl-tmp/output_ours/pruning_target/{method}/{model_struct}/{dataset}.log', mode="w") as f:
            f.write(f"pruning target:{layer_id}")
    if scoreSC < threshold:
        print(f"current score: {scoreSC} < threshold: {threshold}, stop pruning")
        break_alalyse = True
    return break_alalyse, scoreSC

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='oxford_iiit_pet')
    parser.add_argument('--method', type=str, default='convpass', choices=['convpass', 'repadapter'])
    parser.add_argument('--model_struct', type=str, default='ViT', choices=['ViT', 'Swin'])
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    
    model_depth = 12 if args.model_struct == 'ViT' else 24
    best_score = 0
    for i in range(model_depth):
        run_t_SNE(method=args.method, model_struct=args.model_struct, dataset=args.dataset, layer_id=model_depth - i,
                  model_depth=model_depth)
        brear_or_not, score = analyse_t_SNE(method=args.method, model_struct=args.model_struct, dataset=args.dataset,
                                            layer_id=model_depth - i, best_score=best_score)
        if i == 0:
            best_score = score
            print('best_score = ', best_score)
        if brear_or_not:
            break
