import time
from argparse import ArgumentParser

from avalanche.evaluation.metrics.accuracy import Accuracy
from timm import create_model
import torch
from tqdm import tqdm
from thop import profile
from vtab import get_data
from convpass import set_Convpass
from utils import get_config, set_seed
from repadapter import set_RepAdapter, set_RepWeight
from pruning_train import load
from set_pruning import set_drop_backbone


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in tqdm(dl):  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)
    return acc.result()[0]


def vit_forward_drop(self, x):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    for i in range(self.current_depth):
        x = self.blocks[i](x)

    x = self.norm(x)

    if self.dist_token is None:
        return self.pre_logits(x[:, 0])
    else:
        return x[:, 0], x[:, 1]


def swin_forward_drop(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    if self.current_depth >= 23:  # 1->24 2,2,18,2
        # stage4
        for index in range(3):  # stage1,2,3
            x = self.layers[index](x)
        layer_depth = self.current_depth - 22
        x = self.layers[3](x, layer_depth)  # stage4

    elif 22 >= self.current_depth >= 5:
        # stage3
        for index in range(2):  # stage1,2
            x = self.layers[index](x)
        layer_depth = self.current_depth - 4
        x = self.layers[2](x, layer_depth)  # stage3

    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x


def get_acc(model, method, dataset, current_depth, model_struct):
    config = get_config(method=method, dataset_name=dataset)
    if model_struct == 'Pruned':
        train_dl, test_dl = get_data(dataset, evaluate=True)
        model.cuda()
        throughput(model)
        print(test(model, test_dl))
    else:
        if method == 'convpass':
            set_Convpass(model, method, dim=8, s=config['scale'], xavier_init=config['xavier_init'])
        elif method == 'repadapter':
            set_RepAdapter(model, 'repblock', dim=8, s=config['scale'], set_forward=False)
        else:
            raise NotImplementedError
        set_drop_backbone(model, current_depth=current_depth)
        model.reset_classifier(config['class_num'])
        model = load(dataset=dataset, model=model, method=method, model_struct=model_struct, blocks=current_depth)
        train_dl, test_dl = get_data(dataset, evaluate=True)
        if method == 'repadapter':
            set_RepWeight(model, 'repblock', dim=8, s=config['scale'])
        model.cuda()
        throughput(model)
        print(test(model, test_dl))


@torch.no_grad()
def throughput(model, img_size=224, bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size = x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model = model.to('cuda')
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(100):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {100 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)
    return 100 * batch_size / (tic2 - tic1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='oxford_iiit_pet')
    parser.add_argument('--method', type=str, default='convpass', choices=['convpass', 'repadapter'])
    parser.add_argument('--model_struct', type=str, default='ViT', choices=['ViT', 'Swin'])
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    if args.model_struct == 'ViT':
        model = create_model('vit_base_patch16_224_in21k', checkpoint_path='/root/autodl-tmp/pre_train/ViT-B_16.npz',
                             drop_path_rate=0.1)
    elif args.model_struct == 'Swin':
        model = create_model('swin_base_patch4_window7_224_in22k', drop_path_rate=0.1)
        check_point = torch.load('/root/autodl-tmp/pre_train/swin_base_patch4_window7_224_22k.pth')
        model.load_state_dict(check_point['model'])

    config = get_config(args.method, args.dataset)
    args.class_num = config['class_num']
    if args.method == 'convpass':
        set_Convpass(model, 'convpass', dim=8, s=config['scale'], xavier_init=config['xavier_init'])
    elif args.method == 'repadapter':
        set_RepAdapter(model, 'repblock', dim=8, s=config['scale'])

    set_drop_backbone(model, current_depth=9)
    model.reset_classifier(config['class_num'])

    model_throughput = 0
    for i in tqdm(range(100)):
        model_throughput += throughput(model)

    print(f'100 times model throughput result:{model_throughput / 100}')

    # get_acc(model=model, method=args.method, dataset=args.dataset, current_depth=8, model_struct=args.model_struct)
