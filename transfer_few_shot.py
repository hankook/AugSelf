import random
from argparse import ArgumentParser
from functools import partial
from copy import deepcopy
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist

from datasets import load_fewshot_datasets
from models import load_backbone, load_mlp
from trainers import collect_features, SSObjective
from utils import Logger
from transforms import extract_diff

from sklearn.linear_model import LogisticRegression


class FewShotBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, N, K, Q, num_iterations):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_iterations = num_iterations

        labels = [label for _, label in dataset.samples]
        self.label2idx = defaultdict(list)
        for i, y in enumerate(labels):
            self.label2idx[y].append(i)

        few_labels = [y for y, indices in self.label2idx.items() if len(indices) <= self.K]
        for y in few_labels:
            del self.label2idx[y]

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        label_set = set(list(self.label2idx.keys()))
        for _ in range(self.num_iterations):
            labels = random.sample(label_set, self.N)
            indices = []
            for y in labels:
                if len(self.label2idx[y]) >= self.K+self.Q:
                    indices.extend(list(random.sample(self.label2idx[y], self.K+self.Q)))
                else:
                    tmp_indices = [i for i in self.label2idx[y]]
                    random.shuffle(tmp_indices)
                    indices.extend(tmp_indices[:self.K] + np.random.choice(tmp_indices[self.K:], size=self.Q).tolist())
            yield indices


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()
    logger = Logger(None)

    # DATASETS
    datasets = load_fewshot_datasets(dataset=args.dataset,
                                     datadir=args.datadir,
                                     pretrain_data=args.pretrain_data)
    build_sampler    = partial(FewShotBatchSampler,
                               N=args.N, K=args.K, Q=args.Q, num_iterations=args.num_tasks)
    build_dataloader = partial(torch.utils.data.DataLoader,
                               num_workers=args.num_workers)
    testloader  = build_dataloader(datasets['test'],  batch_sampler=build_sampler(datasets['test']))

    # MODELS
    ckpt = torch.load(args.ckpt, map_location=device)
    backbone = load_backbone(args).to(device)
    backbone.load_state_dict(ckpt['backbone'])
    backbone.eval()

    all_accuracies = []
    for i, (batch, _) in enumerate(testloader):
        with torch.no_grad():
            batch = batch.to(device)
            B, C, H, W = batch.shape
            batch = batch.view(args.N, args.K+args.Q, C, H, W)

            train_batch  = batch[:, :args.K].reshape(args.N*args.K, C, H, W)
            test_batch   = batch[:, args.K:].reshape(args.N*args.Q, C, H, W)
            train_labels = torch.arange(args.N).unsqueeze(1).repeat(1, args.K).to(device).view(-1)
            test_labels  = torch.arange(args.N).unsqueeze(1).repeat(1, args.Q).to(device).view(-1)

        with torch.no_grad():
            X_train = backbone(train_batch)
            Y_train = train_labels

            X_test = backbone(test_batch)
            Y_test = test_labels

        classifier = LogisticRegression(solver='liblinear').fit(X_train.cpu().numpy(),
                                                                Y_train.cpu().numpy())
        preds = classifier.predict(X_test.cpu().numpy())
        acc = np.mean((Y_test.cpu().numpy() == preds).astype(float))
        all_accuracies.append(acc)
        if (i+1) % 10 == 0:
            logger.log_msg(f'{i+1:3d} | {acc:.4f} (mean: {np.mean(all_accuracies):.4f})')

    avg = np.mean(all_accuracies)
    std = np.std(all_accuracies) * 1.96 / np.sqrt(len(all_accuracies))
    logger.log_msg(f'mean: {avg:.4f}±{std:.4f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, default='cub200')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num-tasks', type=int, default=2000)
    args = parser.parse_args()
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(None) as parallel:
        parallel.run(main, args)

