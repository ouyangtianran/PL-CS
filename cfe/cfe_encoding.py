import torch
from utils.network import ModelConvMiniImagenet as FeaModelImagenet, ModelConvOmniglotMLP as FeaModelOmniglot
from utils.omniglot import OmniglotConcatDataset, OmniglotCacheDataset
from utils.miniimagenet import MiniImagenetConcatDataset
from utils.tieredimagenet import TieredImagenetConcatDataset

from torchvision.transforms import ToTensor, Resize, Compose, RandomResizedCrop, RandomRotation
import os
import numpy as np
import argparse
import time
from sklearn.cluster import KMeans as SK
from sklearn.preprocessing import StandardScaler
import torchvision.models as models
import torch.nn as nn

parser = argparse.ArgumentParser(description='Extract embedding for few shot dataset')
parser.add_argument('-p', '--pretrain-path', default='../pretrain/omni-embedding.pth.tar', type=str, metavar='S',
                    help='path to the pretrained model')
parser.add_argument('-d', '--dataset', default='omniglot', type=str, metavar='S',
                    help='name of dataset (omniglot, miniimagenet, tieredimagenet)')
parser.add_argument('-s', '--split', default='train', type=str, metavar='S',
                    help='name of subset (train, val, test)')

class Sequential_2output(nn.Sequential):
    def __init__(self, *args):
        super(Sequential_2output, self).__init__(*args)
    def forward(self, input):
        for i, module in enumerate(self):
            input = module(input)
            if i == len(self)-2:
                input_second_last = input
        return input, input_second_last
    
def cluster_and_supply(pos, x1, y1, z1, K):
    # kmeans
    print("=> start kmeans")
    start = time.time()
    X = z1.numpy()
    k_m = SK(n_clusters=K, random_state=0, verbose=1, init='k-means++',
            n_init=10, max_iter=3000).fit(X)
    # k_m = SK(n_clusters=K, random_state=None, verbose=1, init='k-means++',
    #          n_init=1, max_iter=3000).fit(X)
    cl, c = k_m.labels_, k_m.cluster_centers_  # nx1 nxd
    end = time.time()
    print('Timing for kmeans: {:.5f}s \n'.format(end - start))

    cl_u, counts = np.unique(cl, return_counts=True)
    print('the max len of cluster is set as {} the min len of cluster is {}'.format(max(counts), min(counts)))

    x1, y1, z1 = x1.numpy(), y1.numpy(), z1.numpy()
    n_min = 20.0
    for cl0, count in zip(cl_u, counts):
        if count < n_min:
            n_repeat = np.ceil(n_min / count)
            ind = cl == cl0  
            cl00 = np.repeat(cl[ind], n_repeat, axis=0)
            x00 = np.repeat(x1[ind], n_repeat, axis=0)
            y00 = np.repeat(y1[ind], n_repeat, axis=0)
            z00 = np.repeat(z1[ind], n_repeat, axis=0)
            # x200 = np.repeat(x2[ind], n_repeat, axis=0)
            cl = np.concatenate((cl, cl00))
            x1 = np.concatenate((x1, x00))
            y1 = np.concatenate((y1, y00))
            z1 = np.concatenate((z1, z00))
            # x2 = np.concatenate((x2, x200))
    print('the size of samples:{}'.format(x1.shape))
    return cl, c, x1, y1, z1

def cluster_and_save(x1, y1, z1, K, out, dataset, encoding_dim, split, no, random_state=0, n_init=10):
    # kmeans
    print("=> start kmeans")
    start = time.time()
    k_m1 = SK(n_clusters=K, random_state=random_state, verbose=1, init='k-means++', n_init=n_init, max_iter=3000).fit(z1)
    labels, cluster_centers = k_m1.labels_, k_m1.cluster_centers_  # nx1 nxd
    end = time.time()
    print('Timing for kmeans: {:.5f}s \n'.format(end - start))

    np.savez(out + '{}_{}_K_{}_{}_clusters_{}.npz'.format(dataset, encoding_dim, K, split, no),cluster_centers=cluster_centers)
    np.savez(out + '{}_{}_K_{}_{}_{}.npz'.format(dataset, encoding_dim, K, split, no),X=x1, Y=y1, cluster_label=labels)
    np.savez(out + '{}_{}_K_{}_{}_emb_{}.npz'.format(dataset, encoding_dim, K, split, no), Z=z1, cluster_label=labels)
    print('{} save complete \n'.format(no))
    
def cluster_supply_save(x1, y1, z1, K, out, dataset, encoding_dim, split, no, random_state=0, n_init=10):
    # kmeans
    print("=> start kmeans")
    start = time.time()
    k_m1 = SK(n_clusters=K, random_state=random_state, verbose=1, init='k-means++', n_init=n_init, max_iter=3000).fit(z1)
    labels, cluster_centers = k_m1.labels_, k_m1.cluster_centers_  # nx1 nxd
    end = time.time()
    print('Timing for kmeans: {:.5f}s \n'.format(end - start))
    
    cl_u, counts = np.unique(labels, return_counts=True)
    print('the max len of cluster is set as {} the min len of cluster is {}'.format(max(counts), min(counts)))
    # print('the number of cluster: {}, cluster labels: {}'.format(len(cl_u), cl_u))

    # x1, y1, z1 = x1.numpy(), y1.numpy(), z1.numpy()
    pos = np.arange(y1.shape[0])
    n_min = 20.0
    for cl0, count in zip(cl_u, counts):
        if count < n_min:
            n_repeat = np.ceil(n_min / count)
            ind = labels == cl0  
            cl00 = np.repeat(labels[ind], n_repeat, axis=0)
            x00 = np.repeat(x1[ind], n_repeat, axis=0)
            y00 = np.repeat(y1[ind], n_repeat, axis=0)
            z00 = np.repeat(z1[ind], n_repeat, axis=0)
            pos00 = np.repeat(pos[ind], n_repeat, axis=0)
            # x200 = np.repeat(x2[ind], n_repeat, axis=0)
            labels = np.concatenate((labels, cl00))
            x1 = np.concatenate((x1, x00))
            y1 = np.concatenate((y1, y00))
            z1 = np.concatenate((z1, z00))
            pos = np.concatenate((pos, pos00))
            # x2 = np.concatenate((x2, x200))
    print('the size of samples:{}'.format(x1.shape))

    np.savez(out + '{}_{}_K_{}_{}_clusters_{}.npz'.format(dataset, encoding_dim, K, split, no),cluster_centers=cluster_centers)
    np.savez(out + '{}_{}_K_{}_{}_{}.npz'.format(dataset, encoding_dim, K, split, no),N=pos,X=x1, Y=y1, cluster_label=labels)
    np.savez(out + '{}_{}_K_{}_{}_emb_{}.npz'.format(dataset, encoding_dim, K, split, no),N=pos, Z=z1, cluster_label=labels)
    print('{} save complete \n'.format(no))

def get_cfe_fea_1(split, pretrain_path, dataset, supply = True):
    encoder = 'cfe'
    if dataset == 'omniglot':
        encoding_dim = 64
        hidden_size = 64
        fea_model = FeaModelOmniglot(encoding_dim, hidden_size=hidden_size)
        # load dataset
        transform = Compose([Resize(28), ToTensor()])
        if split == 'train':
            meta_train_dataset = OmniglotCacheDataset('./data', meta_train=True, transform=transform)
        elif split == 'val':
            meta_train_dataset = OmniglotCacheDataset('./data', meta_val=True, transform=transform)
        elif split == 'test':
            meta_train_dataset = OmniglotCacheDataset('./data', meta_test=True, transform=transform)
    elif dataset == 'miniimagenet':
        encoding_dim = 128
        model = models.__dict__['resnet50'](num_classes=encoding_dim)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = Sequential_2output(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
        fea_model = model
        # load dataset
        transform = Compose([Resize(84), ToTensor()])
        if split == 'train':
            meta_train_dataset = MiniImagenetConcatDataset('./data', meta_train=True, transform=transform,
                                                       download=True)
        elif split == 'val':
            meta_train_dataset = MiniImagenetConcatDataset('./data', meta_val=True, transform=transform,
                                                       download=True)
        elif split == 'test':
            meta_train_dataset = MiniImagenetConcatDataset('./data', meta_test=True, transform=transform,
                                                       download=True)
    elif dataset == 'tieredimagenet':
        encoding_dim = 128
        model = models.__dict__['resnet50'](num_classes=encoding_dim)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = Sequential_2output(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
        fea_model = model
        # load dataset
        transform = Compose([Resize(84), ToTensor()])
        if split == 'train':
            meta_train_dataset = TieredImagenetConcatDataset('./data', meta_train=True, transform=transform,
                                                       download=True)
        elif split == 'val':
            meta_train_dataset = TieredImagenetConcatDataset('./data', meta_val=True, transform=transform,
                                                       download=True)
        elif split == 'test':
            meta_train_dataset = TieredImagenetConcatDataset('./data', meta_test=True, transform=transform,
                                                       download=True)

    dataloader = torch.utils.data.DataLoader(meta_train_dataset,
                                             batch_size=32, shuffle=False, num_workers=4,
                                             pin_memory=True)
    out = './data/{}_encodings/supply/'.format(encoder)
    if not os.path.exists(out):
        os.mkdir(out)

    if split == 'train':
        K_list=[500]
        if dataset == 'omniglot':  
            if not os.path.exists(out + '{}_{}_xyz.npz'.format(dataset, encoding_dim)):
                print("=> loading checkpoint '{}'".format(pretrain_path))
                checkpoint = torch.load(pretrain_path, map_location="cpu")
                # rename pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                # args.start_epoch = 0
                msg = fea_model.load_state_dict(state_dict, strict=False)
                print("=> loaded pre-trained model '{}'".format(pretrain_path))

                # embedding
                print("=> start encoding")
                fea_model.eval().cuda()
                count = 0
                start = time.time()
                for batch in dataloader:
                    # if count > 1200:
                    #     break
                    with torch.set_grad_enabled(False):
                        x, y = batch[0], batch[1]
                        x = x.detach()
                        new_z= fea_model(x.cuda())
                        if count==0:
                            x1,y1,z1= x,y,new_z.cpu()
                        else:
                            x1,y1,z1= torch.cat((x1,x)),  torch.cat((y1,y)), torch.cat((z1,new_z.cpu()))
                        count += 1
                        if count % 100 ==0:
                            print('count: ', count)
                print('the size of samples:{}'.format(x1.shape))
                end = time.time()
                print('Timing for encoding: {:.5f}s \n'.format(end - start))
                np.savez(out + '{}_{}_xyz.npz'.format(dataset, encoding_dim),x=x1,y=y1,z1=z1)
            else:
                xyz = np.load(out + '{}_{}_xyz.npz'.format(dataset, encoding_dim))
                x1, y1, z1 = xyz['x'], xyz['y'], xyz['z1']  
                
            if supply:
                for K in K_list:
                    no1 = int(time.time()) 
                    cluster_supply_save(x1, y1, z1, K, out, dataset, encoding_dim, split, no1)
            else:
                for K in K_list:
                    no1 = int(time.time()) 
                    cluster_and_save(x1, y1, z1, K, out, dataset, encoding_dim, split, no1)
                
        else: 
            if not os.path.exists(out + '{}_{}_xyz1z2.npz'.format(dataset, encoding_dim)):
                print("=> loading checkpoint '{}'".format(pretrain_path))
                checkpoint = torch.load(pretrain_path, map_location="cpu")
                # rename pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                # args.start_epoch = 0
                msg = fea_model.load_state_dict(state_dict, strict=False)
                print("=> loaded pre-trained model '{}'".format(pretrain_path))

                # embedding
                print("=> start encoding")
                fea_model.eval().cuda()
                count = 0
                start = time.time()
                for batch in dataloader:
                    # if count > 1200:
                    #     break
                    with torch.set_grad_enabled(False):
                        x, y = batch[0], batch[1]
                        x = x.detach()
                        new_z1, new_z2 = fea_model(x.cuda())
                        if count==0:
                            x1,y1,z11,z12 = x,y,new_z1.cpu(), new_z2.cpu()
                        else:
                            x1,y1,z11,z12 = torch.cat((x1,x)),  torch.cat((y1,y)), torch.cat((z11,new_z1.cpu())), torch.cat((z12,new_z2.cpu()))
                        count += 1
                        if count % 100 ==0:
                            print('count: ', count)
                print('the size of samples:{}'.format(x1.shape))
                end = time.time()
                print('Timing for encoding: {:.5f}s \n'.format(end - start))
                np.savez(out + '{}_{}_xyz1z2.npz'.format(dataset, encoding_dim),x=x1,y=y1,z1=z11,z2=z12)
            else:
                xyz1z2 = np.load(out + '{}_{}_xyz1z2.npz'.format(dataset, encoding_dim))
                x1, y1, z11, z12 = xyz1z2['x'], xyz1z2['y'], xyz1z2['z1'], xyz1z2['z2']               
                          
            if supply:
                for K in K_list:
                    no1 = int(time.time()) 
                    cluster_supply_save(x1, y1, z11, K, out, dataset, encoding_dim, split, no1)
                    no2 = int(time.time())  
                    cluster_supply_save(x1, y1, z12, K, out, dataset, encoding_dim, split, no2, random_state=None, n_init=1)
            else:
                for K in K_list:
                    no1 = int(time.time()) 
                    cluster_and_save(x1, y1, z11, K, out, dataset, encoding_dim, split, no1)
                    no2 = int(time.time())  
                    cluster_and_save(x1, y1, z12, K, out, dataset, encoding_dim, split, no2, random_state=None, n_init=1)

    else:
        K = 500
        count = 0
        start = time.time()
        for batch in dataloader:
            with torch.set_grad_enabled(False):
                x, y = batch
                x = x.detach()
                if count == 0:
                    x1, y1 = x, y
                else:
                    x1, y1 = torch.cat((x1, x)), torch.cat((y1, y))
                count += 1
                if count % 20 == 0:
                    print('count: ', count)
        print('the size of samples:{}'.format(x1.shape))
        end = time.time()
        print('Timing for transform: {:.5f}s \n'.format(end - start))
        np.savez(out + '{}_{}_K_{}_{}.npz'.format(dataset, encoding_dim, K, split), X=x1.numpy(), Y=y1.numpy(),
                 cluster_label=y1.numpy())

def main():
    args = parser.parse_args()
    # split = 'train' #splits = ['train', 'val', 'test']
    split = args.split
    pretrain_path = args.pretrain_path
    dataset = args.dataset
    get_cfe_fea_1(split, pretrain_path, dataset)

if __name__ == '__main__':
    main()