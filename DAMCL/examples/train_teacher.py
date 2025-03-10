# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.stats import gaussian_kde
import scipy.spatial.distance as dist
import sys

import collections
import time
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os
from sklearn.cluster import DBSCAN
from torchvision.transforms import InterpolationMode

# from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
import faiss

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from damcl import datasets
from damcl import models
from damcl.models.cm import ClusterMemory, ClusterMemory_teacher, HybridMemory
from damcl.trainers import Trainer_teacher
from damcl.evaluators import Evaluator, extract_features
from damcl.utils.data import IterLoader
from damcl.utils.data import transforms as T
from damcl.utils.data.sampler import RandomMultipleGallerySampler, GroupSampler
from damcl.utils.data.preprocessor import Preprocessor
from damcl.utils.logging import Logger
from damcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from damcl.utils.faiss_rerank import compute_jaccard_distance

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        # T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        # T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    logging_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    sys.stdout = Logger(osp.join(args.logs_dir + "/teacher_model/" + logging_time[:13], 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer_teacher(model, log_loss_path=args.log_loss_path, logging_time=logging_time)

    memory_instance = HybridMemory(model.module.num_features, len(dataset.train),
                                   temp=args.temp, momentum=args.momentum).cuda()
    print("==> Initialize instance features in the hybrid memory")
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _, _, _, _ = extract_features(model, cluster_loader)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    memory_instance.features = F.normalize(features, dim=1).cuda()
    del features

    for epoch in range(args.epochs):

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            median = collections.defaultdict(int)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            for idx in sorted(centers.keys()):
                fea_idx = torch.stack(centers[idx], dim=0).to(device)
                distances = torch.cdist(fea_idx, fea_idx).to("cpu")
                median[idx] = np.argmin(torch.sum(distances, dim=1).cpu())
            centers = [
                torch.stack(centers[idx], dim=0)[median[idx]] for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id != -1:
                    labels.append(id)
                else:
                    labels.append(num + outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            features, features_up, features_down, _, cids = extract_features(model, cluster_loader)

            if epoch == 0:
                print('==> Computing camera label relation matrix...')
                t = time.time()
                cids = torch.cat([cids[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                o = (cids[:, None] == cids[None, :]).int()
                o = np.array(o)
                print('==> camera label relation matrix computing time cost: {}'.format(time.time() - t))

            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, search_option=3, use_float16=False, cam_label=o,
                                                   k1_intra=args.k1_intra, k1_inter=args.k1_inter,
                                                   k2_intra=args.k2_intra, k2_inter=args.k2_inter,
                                                   k1=args.k1, k2=args.k2)

            features_up = torch.cat([features_up[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist_up = compute_jaccard_distance(features_up, search_option=3, use_float16=False, cam_label=o,
                                                      k1_intra=args.k1_intra, k1_inter=args.k1_inter,
                                                      k2_intra=args.k2_intra, k2_inter=args.k2_inter,
                                                      k1=args.k1, k2=args.k2)

            features_down = torch.cat([features_down[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist_down = compute_jaccard_distance(features_down, search_option=3, use_float16=False, cam_label=o,
                                                        k1_intra=args.k1_intra, k1_inter=args.k1_inter,
                                                        k2_intra=args.k2_intra, k2_inter=args.k2_inter,
                                                        k1=args.k1, k2=args.k2)

            rerank_dist = (1.0 - args.lambda1 * 2) * rerank_dist + args.lambda1 * rerank_dist_up + args.lambda1 * rerank_dist_down

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            del rerank_dist, rerank_dist_up,rerank_dist_down

        cluster_features = generate_cluster_features(pseudo_labels, features)
        cluster_features_up = generate_cluster_features(pseudo_labels, features_up)
        cluster_features_down = generate_cluster_features(pseudo_labels, features_down)
        del features, features_up, features_down
        # Create hybrid memory
        memory = ClusterMemory_teacher(model.module.num_features, num_cluster, temp=args.temp,
                                       momentum=args.momentum, use_hard=args.use_hard, lambda2=args.lambda2).cuda()

        memory.features = F.normalize(cluster_features, dim=1).cuda()  
        memory.features_up = F.normalize(cluster_features_up, dim=1).cuda()
        memory.features_down = F.normalize(cluster_features_down, dim=1).cuda()

        # 伪标签数据集（图片地址fname、伪标签、相机id）
        pseudo_labeled_dataset = []
        outliers = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
            else:
                outliers.append((fname, label.item(), cid))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_cluster)

        memory_instance.labels = pseudo_labels.cuda()

        trainer.memory = memory
        trainer.memory_instance = memory_instance

        print('==> Statistics for epoch {}: {} clusters, {} outliers'.format(epoch, num_cluster, len(outliers)))

        trainer.train(args.epochs, epoch, train_loader, optimizer, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            _, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir + "/teacher_model/" + logging_time[:13], 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir + "/teacher_model/" + logging_time[:13], 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    for i in range(1):
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k1-intra', type=int, default=5,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k1-inter', type=int, default=25,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2-intra', type=int, default=2,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2-inter', type=int, default=4,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.0,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,  # default=0.00035
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)  # default=20
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="data")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--log-loss-path', type=str, metavar='PATH',
                        default='logs_loss')
    parser.add_argument('--pooling-type', type=str, default='gem')  # gem
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--enable_tb', action='store_true', help='enable tensorboard logging')
    parser.add_argument('--lambda1', type=float, default=0.15)
    parser.add_argument('--lambda2', type=float, default=0.2)

    main()
