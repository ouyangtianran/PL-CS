import os.path

import torch
import random
import numpy as np
import pandas as pd
import requests
import tqdm

EPSILON = 1e-8
CHUNK_SIZE = 1 * 1024 * 1024


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def load_clusters(dataloader,device):
    # if dataset == 'mini-imagenet':
    #     root = os.path.join(root, 'cfe_encodings/mini_imagenet_128_K_500_train_clusters.npz')
    # elif root=='omniglot':
    #     root = os.path.join(root, 'cfe_encodings/omniglot_64_K_500_train_clusters.npz')
    # elif root=='tiered-imagenet':
    #     root = os.path.join(root, 'cfe_encodings/tiered_imagenet_128_K_500_train_clusters.npz')
    root = dataloader.dataset.dataset.root
    filename = dataloader.dataset.dataset.filename
    cluster_root = os.path.join(root, filename % 'train_clusters')
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(cluster_root)
    np.load = np_load_old
    c_list = dataloader.dataset.dataset.data
    c_centers = data['cluster_centers']
    c = torch.tensor(c_centers).cuda(device=device, non_blocking=True)

    dist = pairwise_distances(c, c, 'l2')
    _, ind_sorted = torch.sort(dist, dim=1)
    return c_list, c_centers, ind_sorted



def momentum_update(model_q, model_k, beta = 0.999, replace=False):
    param_k = model_k.state_dict()
    if replace:
        param_q = model_q.state_dict()
        model_k.load_state_dict(param_q)
    else:
        param_q = model_q.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        model_k.load_state_dict(param_k)
        


def get_last_checkpoint(out_path):
    file_list = os.listdir(out_path)
    checkpoint_list = []
    epoch_list = []
    for file in file_list:
        if file.startswith("model_epoch_"):
            name = file.split('.th')[0]
            epoch = name.split('_')[-1]
            checkpoint_list.append(file)
            epoch_list.append(epoch)
    epoch_max = max(epoch_list)
    max_index = epoch_list.index(epoch_max)
    last_checkpoint = os.path.join(out_path, checkpoint_list[max_index])
    return last_checkpoint
