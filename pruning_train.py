import datetime
import os
import time
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.optim import SGD
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from thop import profile
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import *
from convpass import set_Convpass
from set_pruning import set_drop_backbone, set_limited_convpass, backbone_weight_fusioin, \
    set_mixAdapter, set_Side_Adapter, set_layer_forward, set_block_forward
import nni
import torch_pruning as tp
from nni.utils import merge_parameter
from fine_tune.engine_finetune import train_one_epoch, evaluate
import torch.backends.cudnn as cudnn
from fine_tune.misc import NativeScalerWithGradNormCount as NativeScaler
import fine_tune.misc as misc
import math
import json

from repadapter import set_RepAdapter


@torch.no_grad()
def save(dataset, model, method, model_struct, blocks=12):
    model.eval()
    save_model = model.cpu()
    trainable = {}
    for n, p in save_model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, f"/root/autodl-tmp/output_ours/{method}/{model_struct}/{dataset}_{blocks}blocks.pt")


def load(dataset, model, method, model_struct, blocks=12):
    model = model.cpu()
    print("loading:" + f"/root/autodl-tmp/output_ours/{method}/{model_struct}/{dataset}_{blocks}blocks.pt")
    st = torch.load(f"/root/autodl-tmp/output_ours/{method}/{model_struct}/{dataset}_{blocks}blocks.pt")
    model.load_state_dict(st, False)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--insert_scale', type=float, default=10)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--class_num', type=int, default=100)
    parser.add_argument('--turnon_distill', default=False)
    parser.add_argument('--distill_rate', type=float, default=5e-3)
    parser.add_argument('--fulltune', default=False)
    parser.add_argument('--device', default='cuda')
    # swin_base_patch4_window7_224_in22k
    # vit_base_patch16_224_in21k
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='oxford_flowers102')
    parser.add_argument('--method', type=str, default='convpass', choices=['convpass', 'repadapter'])
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/output_convpass')
    parser.add_argument('--model_struct', type=str, default='ViT', choices=['ViT', 'Swin'])
    parser.add_argument('--pruning_strategy', type=str, default='directly', choices=['directly', 'iteratively'])
    parser.add_argument('--retraining_strategy', type=str, default='FT', choices=['CLR', 'FT', 'TFS'])
    parser.add_argument('--turnon_save', type=bool, default=True)

    args = parser.parse_args()

    # /root/autodl-tmp/output_ours/pruning_target/convpass/ViT
    print(args)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = 'cpu'
    cudnn.benchmark = True

    set_seed(args.seed)
    config = get_config(args.method, args.dataset)
    args.class_num = config['class_num']
    # /root/autodl-tmp/pre_train/ViT-B_16.npz
    # /root/autodl-tmp/pre_train/swin_base_patch4_window7_224_22k.pth

    if args.model_struct == 'ViT':
        model = create_model(args.model, checkpoint_path='/root/autodl-tmp/pre_train/ViT-B_16.npz',
                             drop_path_rate=0.1)
    elif args.model_struct == 'Swin':
        model = create_model('swin_base_patch4_window7_224_in22k', drop_path_rate=0.1)
        check_point = torch.load('/root/autodl-tmp/pre_train/swin_base_patch4_window7_224_22k.pth')
        model.load_state_dict(check_point['model'])
    # elif args.model_struct == 'Conv':
    #     # convnext_base_22k_224
    #     model = create_model('convnext_base_22k_224', drop_path_rate=0.1, pretrained=True)
    else:
        raise NotImplementedError

    print(model)
    if args.method == 'convpass':
        set_Convpass(model, 'convpass', dim=8, s=config['scale'], xavier_init=config['xavier_init'])
    elif args.method == 'repadapter':
        set_RepAdapter(model, 'repblock', dim=8, s=config['scale'])

    # get target depth
    with open(f'/root/autodl-tmp/output_ours/pruning_target/{args.method}/{args.model_struct}/{args.dataset}.log',
              mode="r") as f:
        target_depth = int(f.read().split(':')[-1])
    print(target_depth)

    for i in range(1):  # 0-4
        start_time = time.time()

        train_dl, test_dl = get_data(args.dataset, evaluate=True)

        if args.model_struct == 'ViT':
            start_depth = 12
        elif args.model_struct == 'Swin':
            start_depth = 24

        # current_depth = start_depth - i
        current_depth = target_depth
        print(f"fine-tuning {current_depth} block backbone")

        set_drop_backbone(model, current_depth=current_depth)
        model.reset_classifier(config['class_num'])

        # if i != 0 and args.retraining_strategy != 'TFS':
        #     if args.pruning_strategy == 'directly':
        #         model = load(args.dataset, model, args.method, model_struct=args.model_struct, blocks=start_depth)
        #     elif args.pruning_strategy == 'iteratively':
        #         model = load(args.dataset, model, args.method, model_struct=args.model_struct, blocks=current_depth + 1)

        if args.retraining_strategy != 'TFS':
            if args.pruning_strategy == 'directly':
                model = load(args.dataset, model, args.method, model_struct=args.model_struct, blocks=start_depth)
            elif args.pruning_strategy == 'iteratively':
                model = load(args.dataset, model, args.method, model_struct=args.model_struct, blocks=current_depth + 1)

        for name, p in model.named_parameters():
            if 'adapt' in name:
                p.requires_grad = True
                print(name)
            else:
                p.requires_grad = False if not args.fulltune else True

        for _, p in model.head.named_parameters():
            p.requires_grad = True

        model = model.to(device)

        config['best_acc'] = 0
        config['method'] = args.method
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_parameters_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        n_parameters = n_parameters - n_parameters_head

        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        if args.retraining_strategy == 'FT':
            args.lr = args.min_lr
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
                model, criterion, train_dl,
                opt, device, epoch, loss_scaler,
                max_norm=None,
                log_writer=log_writer,
                args=args
            )

            test_stats = evaluate(test_dl, model, device, args=args)
            if args.output_dir and test_stats["acc1"] > max_accuracy:
                if args.turnon_save:
                    save(args.dataset, model, args.method, model_struct=args.model_struct, blocks=current_depth)
            print(f"Accuracy of the network on the {len(test_dl.dataset)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])

            # nni.report_intermediate_result(test_stats["acc1"])

            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
        if args.turnon_save:
            with open(f'/root/autodl-tmp/output_ours/{args.method}/{args.model_struct}/{args.dataset}.log',
                      mode="a") as f:
                f.write(
                    f"{current_depth}blocks {args.pruning_strategy} pruning {args.retraining_strategy} retraining acc:" + str(
                        max_accuracy) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print(config['best_acc'])
