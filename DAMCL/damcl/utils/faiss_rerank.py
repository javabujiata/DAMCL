#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
    index_init_gpu, index_init_cpu


# 相机间差距过大，自适应调整样本的k相互近邻


def k_reciprocal_neigh_cam(rank_intra, rank_inter, i, k1_intra, k1_inter):
    forward_k_neigh_index_intra = rank_intra[i, 1:k1_intra + 1]
    forward_k_neigh_index_inter = rank_inter[i, :k1_inter]
    backward_k_neigh_index_intra = rank_intra[forward_k_neigh_index_intra, 1:k1_intra + 1]
    backward_k_neigh_index_inter = rank_inter[forward_k_neigh_index_inter, :k1_inter]
    fi_intra = np.where(backward_k_neigh_index_intra == i)[0]
    fi_inter = np.where(backward_k_neigh_index_inter == i)[0]
    return forward_k_neigh_index_intra[fi_intra], forward_k_neigh_index_inter[fi_inter]


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, print_flag=True, search_option=0, use_float16=False, k1=30, k2=6,
                             cam_label=None, k1_intra=5, k1_inter=20, k2_intra=2, k2_inter=4):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option == 0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        initial_distance, initial_rank = search_raw_array_pytorch(res, target_features, target_features, N)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option == 1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        initial_distance, initial_rank = search_index_pytorch(index, target_features, N)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option == 2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        initial_distance, initial_rank = index.search(target_features.cpu().numpy(), N)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        # initial_distance, initial_rank = index.search(target_features.cpu().numpy(), 13312)
        initial_distance, initial_rank = index.search(target_features.cpu().numpy(), 6144)

    # intra_masks = [np.isin(initial_rank[i, :], np.where(cam_label[i, :] == 1)[0]) for i in range(N)]
    # rank_intra = [initial_rank[i, mask][:k1] for i, mask in enumerate(intra_masks)]
    # rank_inter = [initial_rank[i, ~mask][:k1] for i, mask in enumerate(intra_masks)]
    # deltai = [np.sum(initial_distance[i, ~mask][:k1]) - np.sum(initial_distance[i, mask][:k1]) for i, mask in
    #           enumerate(intra_masks)]
    intra_masks = [cam_label[i, initial_rank[i, :]] == 1 for i in range(N)]
    rank_intra = [initial_rank[i, mask][:10] for i, mask in enumerate(intra_masks)]
    rank_inter = [initial_rank[i, ~mask][:k1] for i, mask in enumerate(intra_masks)]
    deltai = [np.sum(initial_distance[i, ~mask][:k1])/len(initial_distance[i, ~mask][:k1]) - 
              np.sum(initial_distance[i, mask][:k1])/len(initial_distance[i, mask][:k1]) for i, mask in
              enumerate(intra_masks)]
    rank_intra = np.array(rank_intra)
    rank_inter = np.array(rank_inter)
    del initial_distance, intra_masks
    # nn_k1_intra = []
    # nn_k1_inter = []
    # for i in range(N):
    #     intras, inters = k_reciprocal_neigh(rank_intra, rank_inter, i, k1_intra, k1_inter)
    #     nn_k1_intra.append(intras)
    #     nn_k1_inter.append(inters)
    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        if deltai[i] > 0:
            nn_intra, nn_inter = k_reciprocal_neigh_cam(rank_intra, rank_inter, i, k1_intra, k1_inter)
            k_reciprocal_index = np.unique(np.concatenate((nn_intra, nn_inter), axis=0))
            dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_index].t())
            if use_float16:
                V[i, k_reciprocal_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            else:
                V[i, k_reciprocal_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()
        else:
            k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index
            for candidate in k_reciprocal_expansion_index:
                candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1/2)))
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_expansion_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_index = np.append(k_reciprocal_index, candidate_k_reciprocal_index)
            k_reciprocal_index = np.unique(k_reciprocal_index)
            dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(),
                                    target_features[k_reciprocal_index].t())
            if use_float16:
                V[i, k_reciprocal_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            else:
                V[i, k_reciprocal_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    if k2_intra != 1 and k2_inter != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            if deltai[i] > 0:
                V_qe[i, :] = np.mean(np.concatenate((V[rank_intra[i, :k2_intra], :], V[rank_inter[i, :k2_inter], :])),
                                     axis=0)
            else:
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del rank_inter, rank_intra, initial_rank, deltai

    # invIndex = []
    # for i in range(N):
    #     invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(N)]

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist
