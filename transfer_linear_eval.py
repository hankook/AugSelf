from argparse import ArgumentParser
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist

from datasets import load_datasets
from models import load_backbone
from trainers import collect_features
from utils import Logger


def build_step(X, Y, classifier, optimizer, w):
    def step():
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(X), Y, reduction='sum')
        for p in classifier.parameters():
            loss = loss + p.pow(2).sum().mul(w)
        loss.backward()
        return loss
    return step


def compute_accuracy(X, Y, classifier, metric):
    with torch.no_grad():
        preds = classifier(X).argmax(1)
        if metric == 'top1':
            acc = (preds == Y).float().mean().item()
        elif metric == 'class-avg':
            total, count = 0., 0.
            for y in range(0, Y.max().item()+1):
                masks = Y == y
                if masks.sum() > 0:
                    total += (preds[masks] == y).float().mean().item()
                    count += 1
            acc = total / count
        else:
            raise Exception(f'Unknown metric: {metric}')
    return acc


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()
    logger = Logger(None)

    # DATASETS
    datasets = load_datasets(dataset=args.dataset,
                             datadir=args.datadir,
                             pretrain_data=args.pretrain_data)
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=False)
    valloader   = build_dataloader(datasets['val'],   drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)
    num_classes = datasets['num_classes']

    # MODELS
    ckpt = torch.load(args.ckpt, map_location=device)
    backbone = load_backbone(args)
    backbone.load_state_dict(ckpt['backbone'])

    build_model = partial(idist.auto_model, sync_bn=True)
    backbone   = build_model(backbone)

    # EXTRACT FROZEN FEATURES
    logger.log_msg('collecting features ...')
    X_train, Y_train = collect_features(backbone, trainloader, device, normalize=False)
    X_val,   Y_val   = collect_features(backbone, valloader,   device, normalize=False)
    X_test,  Y_test  = collect_features(backbone, testloader,  device, normalize=False)
    classifier = nn.Linear(args.num_backbone_features, num_classes).to(device)
    optim_kwargs = {
        'line_search_fn': 'strong_wolfe',
        'max_iter': 5000,
        'lr': 1.,
        'tolerance_grad': 1e-10,
        'tolerance_change': 0,
    }
    logger.log_msg('collecting features ... done')

    best_acc = 0.
    best_w = 0.
    best_classifier = None
    for w in torch.logspace(-6, 5, steps=45).tolist():
        optimizer = optim.LBFGS(classifier.parameters(), **optim_kwargs)
        optimizer.step(build_step(X_train, Y_train, classifier, optimizer, w))
        acc = compute_accuracy(X_val, Y_val, classifier, args.metric)

        if best_acc < acc:
            best_acc = acc
            best_w = w
            best_classifier = deepcopy(classifier)

        logger.log_msg(f'w={w:.4e}, acc={acc:.4f}')

    logger.log_msg(f'BEST: w={best_w:.4e}, acc={best_acc:.4f}')

    X = torch.cat([X_train, X_val], 0)
    Y = torch.cat([Y_train, Y_val], 0)
    optimizer = optim.LBFGS(best_classifier.parameters(), **optim_kwargs)
    optimizer.step(build_step(X, Y, best_classifier, optimizer, best_w))
    acc = compute_accuracy(X_test, Y_test, best_classifier, args.metric)
    logger.log_msg(f'test acc={acc:.4f}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--metric', type=str, default='top1')
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)

