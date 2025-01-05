import torch
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import *
from convpass import set_Convpass
from sklearn import manifold
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

def train(config, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            print(f'test accuracy: {acc*100: .2f}%')
            if acc > config['best_acc']:
                config['best_acc'] = acc
                save(config['method'], config['name'], model, acc, ep)
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    model = model.cuda()
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        preds = out.argmax(dim=1).view(-1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def t_SNE(feat):
    # t-SNE
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

def forward_ViT_features(self, x):
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

def pruned_forward(self, x):
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    for i in range(self.pruning_target):
        x = self.blocks[i](x)
    x = self.norm(x)
    return x

def plotlabels(S_lowDWeights, label, args, layer_index):
    print("start plot t_SNE result")
    config = get_config(method=args.method, dataset_name=args.dataset)
    label = label.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, label))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    file_path = f'visualization/{args.method}/{args.dataset}/{layer_index}th_layer.csv'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    S_data.to_csv(file_path)
    
    if args.plot_tsne:

        fig = plt.figure(figsize=(20, 10))
        for index in range(config['class_num']):  #class_num
            X = S_data.loc[S_data['label'] == index]['x']
            Y = S_data.loc[S_data['label'] == index]['y']
            plt.scatter(X, Y, s=100, alpha=0.65)

            plt.xticks([])  
            plt.yticks([])  
            plt.axis('off')
        
        name = f'{args.method}_{args.dataset}_{layer_index}blocks'
        plt.title(name, fontsize=64, fontweight='normal', pad=20)
        plt.show()
        plt.savefig(f'visualization/{args.method}/{args.dataset}/{layer_index}th_layer.png', dpi=300)
    

@torch.no_grad()
def get_feature(model, args, layer_index):
    model.eval()
    feature_collection = []
    label_collection = []
    layer_index = layer_index
    for batch in tqdm(train_dl):
        input, label = batch[0].to('cuda'), batch[1].to('cuda')
        pred, feature_list = forward_ViT_features(model, input)
        feature_collection.append(feature_list[layer_index - 1][:, 0])
        label_collection.append(label)
    intermediate_feature = torch.cat(feature_collection, dim=0)
    label = torch.cat(label_collection, dim=0)
    return intermediate_feature.cpu(), label.cpu()


def run_t_SNE(model, args, layer_index):
    print('start run t_SNE')
    feat, label = get_feature(model, args, layer_index=layer_index)
    plotlabels(t_SNE(feat), label, args, layer_index=layer_index)


def analyse_t_SNE(args, layer_index, best_score):
    print(f'start analyse {layer_index}th_layer t_SNE')
    df = pd.read_csv(
        f'visualization/{args.method}/{args.dataset}/{layer_index}th_layer.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    X = df[['x', 'y']].to_numpy()
    label = df['label'].to_numpy()
    # SC index
    scoreSC = silhouette_score(X, label)

    print(f"{layer_index}th layer")
    print(f"SC Index: {scoreSC}")
    break_alalyse = False
    
    threshold = args.alpha * best_score

    if scoreSC >= threshold:
        print(f"current score:{scoreSC} >= threshold:{threshold}")
        file_path = f'pruning_target/{args.method}/{args.dataset}.log'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, mode="w") as f:
            f.write(f"pruning target:{layer_index}")
    if scoreSC < threshold:
        print(f"current score: {scoreSC} < threshold: {threshold}, stop iteration")
        break_alalyse = True
    return break_alalyse, scoreSC

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='eurosat')
    parser.add_argument('--method', type=str, default='convpass')
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--plot_tsne', type=bool, default=False)
    parser.add_argument('--retraining', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = get_config(args.method, args.dataset)
    model = create_model(args.model, checkpoint_path='ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset)

    set_Convpass(model, args.method, dim=8, s=config['scale'], xavier_init=config['xavier_init'])

    model.reset_classifier(config['class_num'])
    
    load(args.method,args.dataset,model)
    print('testing the original model')
    acc=test(model,test_dl)
    
    print(f'original accuracy: {acc*100: .2f}%')
    
    best_score = 0
    for i in range(12):
        run_t_SNE(model, args, layer_index=12 - i)
        break_or_not, score = analyse_t_SNE(args, layer_index=12 - i, best_score=best_score)
        if i == 0:
            best_score = score
        if break_or_not:
            break

    with open(f'pruning_target/{args.method}/{args.dataset}.log', 'r') as file:
        content = file.read()
    pruning_target = int(content.split(':')[1].strip())

    print(f"Prune to the {pruning_target}th layer:")
    
    model.pruning_target = pruning_target
    bound_method = pruned_forward.__get__(model, model.__class__)
    setattr(model, 'forward_features', bound_method)
    
    acc=test(model,test_dl)
    print(f'pruned accuracy w/o retraining: {acc*100: .2f}%')
    
    if args.retraining:
        
        trainable = []
        config['best_acc'] = 0
        config['method'] = 'pruned_'+args.method
    
        for n, p in model.named_parameters():
            if 'adapter' in n or 'head' in n:
                trainable.append(p)
            else:
                p.requires_grad = False
            
        opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
        
        print('retraining the pruned model')
        model = train(config, model, train_dl, opt, scheduler, epoch=100)
        acc = config['best_acc']
        print(f'pruned accuracy w/ retraining: {acc*100: .2f}%')
