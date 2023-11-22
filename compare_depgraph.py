import os, sys

from timm import create_model
from fine_tune.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.optim import AdamW

from vtab import get_data
from convpass import set_Convpass
from utils import get_config, set_seed
from pruning_train import load
from fine_tune.engine_finetune import train_one_epoch, evaluate

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch.nn.functional as F
import torch_pruning as tp
import timm
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from evaluate import throughput
# import presets

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help='model name')
    parser.add_argument('--data_path', default='data/imagenet', type=str, help='model name')
    parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--pruning_ratio', default=0.3333, type=float, help='prune ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--pruning_type', default='taylor', type=str, help='pruning type',
                        choices=['random', 'taylor', 'l2', 'l1', 'hessian'])
    parser.add_argument('--test_accuracy', default=True, action='store_true', help='test accuracy')
    parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--prune_num_heads', default=False, action='store_true', help='global pruning')
    parser.add_argument('--head_pruning_ratio', default=0.0, type=float, help='head pruning ratio')
    parser.add_argument('--use_imagenet_mean_std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--train_batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
    parser.add_argument('--save_as', default='/root/autodl-tmp/output_ours/pruned_model/', type=str,
                        help='save the pruned model')
    # ours
    parser.add_argument('--dataset', type=str, default='caltech101')
    parser.add_argument('--method', type=str, default='convpass', choices=['convpass', 'repadapter'])
    parser.add_argument('--model_struct', type=str, default='ViT', choices=['ViT', 'Swin'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--insert_scale', type=float, default=10)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--class_num', type=int, default=100)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--ignore_adapter', type=bool, default=True)
    parser.add_argument('--retraining_strategy', type=str, default='CLR', choices=['CLR', 'FT', 'TFS'])
    args = parser.parse_args()
    return args


# Here we re-implement the forward function of timm.models.vision_transformer.Attention
# as the original forward function requires the input and output channels to be identical.
def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1)  # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


# def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4,
#                      use_imagenet_mean_std=False):
#     """The imagenet_root should contain train and val folders.
#     """
#
#     print('Parsing dataset...')
#     train_dst = ImageFolder(os.path.join(imagenet_root, 'train'),
#                             transform=presets.ClassificationPresetEval(
#                                 mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
#                                 std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
#                                 crop_size=224,
#                                 resize_size=256,
#                                 interpolation=InterpolationMode.BILINEAR,
#                             )
#                             )
#     val_dst = ImageFolder(os.path.join(imagenet_root, 'val'),
#                           transform=presets.ClassificationPresetEval(
#                               mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
#                               std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
#                               crop_size=224,
#                               resize_size=256,
#                               interpolation=InterpolationMode.BILINEAR,
#                           )
#                           )
#     train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, shuffle=True,
#                                                num_workers=num_workers)
#     val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
#     return train_loader, val_loader


def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for k, (images, labels) in enumerate(tqdm(val_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    example_inputs = torch.randn(1, 3, 224, 224)

    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == 'l2':
        imp = tp.importance.GroupNormImportance(p=2)
    elif args.pruning_type == 'l1':
        imp = tp.importance.GroupNormImportance(p=1)
    elif args.pruning_type == 'hessian':
        imp = tp.importance.GroupHessianImportance()
    else:
        raise NotImplementedError

    if args.pruning_type in ['taylor', 'hessian'] or args.test_accuracy:
        # train_loader, val_loader = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size,
        #                                             val_batch_size=args.val_batch_size,
        #                                             use_imagenet_mean_std=args.use_imagenet_mean_std)
        train_loader, val_loader = get_data(args.dataset, evaluate=True, batch_size=32)

    # Load the model
    # model = timm.create_model(args.model_name, pretrained=True).eval().to(device)

    config = get_config(args.method, args.dataset)
    args.class_num = config['class_num']
    model = create_model('vit_base_patch16_224_in21k', checkpoint_path='/root/autodl-tmp/pre_train/ViT-B_16.npz',
                         drop_path_rate=0.1)
    set_Convpass(model, 'convpass', dim=8, s=config['scale'], xavier_init=config['xavier_init'])
    model.reset_classifier(config['class_num'])
    model = load(args.dataset, model, args.method, model_struct=args.model_struct, blocks=12)

    # for name, p in model.named_parameters():
    #     if 'adapt' in name:
    #         p.requires_grad = True
    #         print(name)
    #     else:
    #         p.requires_grad = False
    #
    # for _, p in model.head.named_parameters(): 
    #     p.requires_grad = True

    model = model.to(device)

    input_size = [3, 224, 224]
    example_inputs = torch.randn(1, *input_size).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    # throughput(model, bs=64)

    print("Pruning %s..." % args.model_name)
    print(model)
    num_heads = {}
    ignored_layers = [model.head]
    for layer in range(12):
        ignored_layers.append(model.blocks[layer].adapter_attn)
        ignored_layers.append(model.blocks[layer].adapter_mlp)
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            m.forward = forward.__get__(m,
                                        timm.models.vision_transformer.Attention)  # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            num_heads[m.qkv] = m.num_heads
        if args.bottleneck and isinstance(m, timm.models.vision_transformer.Mlp):
            ignored_layers.append(m.fc2)  # only prune the internal layers of FFN & Attention

    print(ignored_layers)
    if args.test_accuracy:
        print("Testing accuracy of the original model...")
        acc_ori, loss_ori = validate_model(model, val_loader, device)
        print("Accuracy: %.4f, Loss: %.4f" % (acc_ori, loss_ori))

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        global_pruning=args.global_pruning,  # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp,  # importance criterion for parameter selection
        pruning_ratio=args.pruning_ratio,  # target pruning ratio
        ignored_layers=ignored_layers,
        num_heads=num_heads,  # number of heads in self attention
        prune_num_heads=args.prune_num_heads,  # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=not args.prune_num_heads,
        # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.5,
        # args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=2
    )

    if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
        model.zero_grad()
        if isinstance(imp, tp.importance.GroupHessianImportance):
            imp.zero_grad()
        print("Accumulating gradients for pruning...")
        for k, (imgs, lbls) in enumerate(train_loader):
            if k >= args.taylor_batchs: break
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            output = model(imgs)
            if isinstance(imp, tp.importance.GroupHessianImportance):
                loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, tp.importance.GroupTaylorImportance):
                loss = torch.nn.functional.cross_entropy(output, lbls)
                loss.backward()

    for i, g in enumerate(pruner.step(interactive=True)):
        g.prune()

    # Modify the attention head size and all head size aftering pruning
    head_id = 0
    for m in model.modules():
        if isinstance(m, timm.models.vision_transformer.Attention):
            print("Head #%d" % head_id)
            print("[Before Pruning] Num Heads: %d, Head Dim: %d =>" % (m.num_heads, m.head_dim))
            m.num_heads = pruner.num_heads[m.qkv]
            m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            print("[After Pruning] Num Heads: %d, Head Dim: %d" % (m.num_heads, m.head_dim))
            print()
            head_id += 1

    print(model)

    if args.test_accuracy:
        print("Testing accuracy of the pruned model...")
        acc_pruned, loss_pruned = validate_model(model, val_loader, device)
        print("Accuracy: %.4f, Loss: %.4f" % (acc_pruned, loss_pruned))

    print("----------------------------------------")
    print("Summary:")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("Base MACs: %.2f G, Pruned MACs: %.2f G" % (base_macs / 1e9, pruned_macs / 1e9))
    print("Base Params: %.2f M, Pruned Params: %.2f M" % (base_params / 1e6, pruned_params / 1e6))

    # throughput(model)
    if args.test_accuracy:
        print("Base Loss: %.4f, Pruned Loss: %.4f" % (loss_ori, loss_pruned))
        print("Base Accuracy: %.4f, Pruned Accuracy: %.4f" % (acc_ori, acc_pruned))

    # retrain
    if args.retrain:
        train_loader, val_loader = get_data(args.dataset, evaluate=True, batch_size=64)
        for name, p in model.named_parameters():
            if 'adapt' in name:
                p.requires_grad = True
                print(name)
            else:
                p.requires_grad = False

        for _, p in model.head.named_parameters(): 
            p.requires_grad = True

        opt = AdamW([p for name, p in model.named_parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

        log_writer = None
        loss_scaler = NativeScaler()
        criterion = torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(criterion))
        print(f"Start training for {args.epochs} epochs")
        max_accuracy = 0.0

        for epoch in range(args.start_epoch, args.epochs):  # args.start_epoch->args.epochs-1
            model = model.to(device)
            train_stats = train_one_epoch(
                model, criterion, train_loader,
                opt, device, epoch, loss_scaler,
                max_norm=None,
                log_writer=log_writer,
                args=args
            )
            test_stats = evaluate(val_loader, model, device, args=args)
            print(f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')


if __name__ == '__main__':
    main()
