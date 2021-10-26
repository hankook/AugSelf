# Improving Transferability of Representations via Augmentation-Aware Self-Supervision

Accepted to NeurIPS 2021

<p align="center">
<img width="762" alt="thumbnail" src="https://user-images.githubusercontent.com/4075389/138967888-29208bbe-d9e7-4bc7-b0b6-15ecbd5d277c.png">
</p>

**TL;DR:** Learning augmentation-aware information by predicting the difference between two augmented samples improves the transferability of representations.

## Dependencies

```bash
conda create -n AugSelf python=3.8 pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 ignite -c pytorch
conda activate AugSelf
pip install scipy tensorboard kornia==0.4.1 sklearn
```

## Checkpoints

We provide ImageNet100-pretrained models in `checkpoints/`.

## Pretraining

We here provide SimSiam+AugSelf pretraining scripts. For training the baseline (i.e., no AugSelf), remove `--ss-crop` and `--ss-color` options. For using other frameworks like SimCLR, use the `--framework` option.

### STL-10
```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    --logdir ./logs/stl10/simsiam/aug_self \
    --framework simsiam \
    --dataset stl10 \
    --datadir DATADIR \
    --model resnet18 \
    --batch-size 256 \
    --max-epochs 200 \
    --ss-color 1.0 --ss-crop 1.0
```

### ImageNet100

```bash
python pretrain.py \
    --logdir ./logs/imagenet100/simsiam/aug_self \
    --framework simsiam \
    --dataset imagenet100 \
    --datadir DATADIR \
    --batch-size 256 \
    --max-epochs 500 \
    --model resnet50 \
    --base-lr 0.05 --wd 1e-4 \
    --ckpt-freq 50 --eval-freq 50 \
    --ss-crop 0.5 --ss-color 0.5 \
    --num-workers 16 --distributed
```

## Evaluation

Our main evaluation setups are linear evaluation on fine-grained classification datasets (Table 1) and few-shot benchmarks (Table 2).

### linear evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python transfer_linear_eval.py \
    --pretrain-data imagenet100 \
    --ckpt CKPT \
    --model resnet50 \
    --dataset cifar10 \
    --datadir DATADIR \
    --metric top1
```

### few-shot

```bash
CUDA_VISIBLE_DEVICES=0 python transfer_few_shot.py \
    --pretrain-data imagenet100 \
    --ckpt CKPT \
    --model resnet50 \
    --dataset cub200 \
    --datadir DATADIR
```
