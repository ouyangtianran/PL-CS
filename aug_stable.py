import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy.stats import entropy

from plot import get_clustered_list, get_clustered_list_1, store_hist_pic, get_clu_pos_list
from cfe.utils.miniimagenet import MiniImagenetConcatDataset
from cfe.utils.omniglot import OmniglotConcatDataset, OmniglotCacheDataset
from cfe.utils.tieredimagenet import TieredImagenetConcatDataset
from cfe.utils.network import ModelConvOmniglotMLP as FeaModelOmniglot
from maml.utils import tensors_to_device

import cfe.loader

def load_model_state(pretrain_path):
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
    return state_dict

def get_aug_stable(embs, aug_emb, clu_center, clu_pos_list, K=0):
    pos = embs['N']
    raw_emb = embs['Z']
    label = embs['cluster_label']
    center = clu_center['cluster_centers']
    aug_emb_supply = aug_emb[pos]
    
    if K==0:
        # clu_center distance
        estimator = KNeighborsClassifier(n_neighbors=1)
        center_label = range(len(center))
        estimator.fit(center, center_label)
        aug_label = estimator.predict(aug_emb_supply)
    else:
        # KNN
        estimator = KNeighborsClassifier(n_neighbors=K)
        estimator.fit(raw_emb, label)
        aug_label = estimator.predict(aug_emb_supply)
    
    stable = label == aug_label
    stable_ratio = sum(stable)/len(label) 
    # 每簇的增强后稳定性
    clu_stability_list = []
    clu_aug_label_list = []
    clu_num = len(clu_pos_list)
    clu_aug_label_matrix = np.zeros((clu_num, clu_num))
    for i, clu_pos in enumerate(clu_pos_list):
        clu_stability = stable[list(clu_pos)]
        clu_stability_list.append((sum(clu_stability)/len(clu_pos), sum(clu_stability),len(clu_pos)))
        
        aug_l, counts = np.unique(aug_label[list(clu_pos)], return_counts=True)
        clu_aug_label_list.append((aug_l,counts,counts/sum(counts)))
        for l, count in zip(aug_l, counts):
            clu_aug_label_matrix[i][int(l)]= int(count)/sum(counts)
        
    return clu_aug_label_matrix, clu_aug_label_list, clu_stability_list, stable, stable_ratio, aug_emb_supply, aug_label

def get_clu_entropy(clu_Y_list, label_dis_entro_dict):
    clu_entropy_list = []
    clu_num_list = []
    clu_purity_list = []
    clu_mainL_dis_entro_list = []
    clu_dis_entro_list = []
    for clu_y in clu_Y_list:
        clu_y_uni, counts = np.unique(clu_y, return_counts=True)
        probs = counts/len(counts)
        clu_entropy_list.append(entropy(probs))
        clu_num_list.append(len(clu_y_uni))
        clu_purity_list.append(max(counts)/(len(clu_y)))
        clu_mainL_dis_entro_list.append(label_dis_entro_dict[clu_y_uni[np.argmax(counts)]])
        dis_entro = 0
        for i, l in enumerate(clu_y_uni):
            dis_entro += probs[i]*label_dis_entro_dict[l]
        clu_dis_entro_list.append(dis_entro)           
    return clu_entropy_list, clu_num_list, clu_purity_list, clu_mainL_dis_entro_list, clu_dis_entro_list

def get_label_dis_entro(clu_Y_list):
    label_dis_dict = {}
    label_dis_entro_dict = {}
    for clu_y in clu_Y_list:
        clu_y_uni, counts = np.unique(clu_y, return_counts=True)
        for y,count in zip(clu_y_uni,counts):
            if y not in label_dis_dict.keys():
                label_dis_dict[y] = []
            label_dis_dict[y].append(count)
    for y,count_list in label_dis_dict.items():
        count_arr = np.asarray(count_list)
        probs = count_arr/len(count_list)
        label_dis_entro_dict[y] = entropy(probs)
    return label_dis_entro_dict

def encode_aug_z(pretrain_path, dataset = 'miniimagenet', weak = False, device = None): 
    if dataset == 'omniglot':
        encoding_dim = 64
        hidden_size = 64
        fea_model = FeaModelOmniglot(encoding_dim, hidden_size=hidden_size)
        if not weak:
            transform = transforms.Compose([
                transforms.Resize(28),
                # transforms.RandomRotation(20),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.1), ratio=(0.9, 1.1), interpolation=2),
                transforms.RandomApply([cfe.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(28),
                # transforms.RandomRotation(20),
                # transforms.RandomResizedCrop(28, scale=(0.2, 1.1), ratio=(0.9, 1.1), interpolation=2),  # weak
                # transforms.RandomResizedCrop(28, scale=(0.6, 1.1), ratio=(0.9, 1.1), interpolation=2), # weak1
                transforms.RandomResizedCrop(28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2), # weak2
                # transforms.RandomApply([cfe.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                ])
        # pretrain_path = "./pretrain/omni-embedding.pth.tar"
        dataset = OmniglotCacheDataset('./data', meta_train=True, transform=transform)
    elif dataset == 'miniimagenet' or "tieredimagenet":
        encoding_dim = 128
        model = models.__dict__['resnet50'](num_classes=encoding_dim)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)  # 在最后的全连接层前加了一层2048*2048的全连接
        fea_model = model
        if not weak:
            transform = transforms.Compose([
                transforms.Resize(84),
                # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([cfe.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(84),
                transforms.RandomResizedCrop(84, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([cfe.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        # pretrain_path = "./pretrain/imagenet-embedding.pth.tar"
        if dataset == 'miniimagenet':
            dataset = MiniImagenetConcatDataset('./data', meta_train=True, transform=transform, download=False)
        else:
            dataset = TieredImagenetConcatDataset('./data', meta_train=True, transform=transform, download=False)
    state_dict=load_model_state(pretrain_path)
    msg = fea_model.load_state_dict(state_dict, strict=False)
    print("=> loaded pre-trained model '{}'".format(pretrain_path))
    
    # data loader 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print("=> start encoding")
    if device is not None:
        fea_model.eval().cuda(device=device)
    else:
        fea_model.eval().cuda()
    count = 0
    for batch in dataloader:
        if device is not None: # gpu
            batch = tensors_to_device(batch, device=device)
            with torch.set_grad_enabled(False):
                x, y = batch[0], batch[1]
                x = x.detach()
                z1 = fea_model(x)
                if count==0:
                    z11 = z1
                else:
                    z11= torch.cat((z11,z1))
                count += 1
                if count % 100 ==0:
                    print('count: ', count)
        else: # cpu
            with torch.set_grad_enabled(False):
                x, y = batch[0], batch[1]
                x = x.detach()
                z1 = fea_model(x.cuda())
                if count==0:
                    x1,y1,z11 = x,y,z1.cpu()
                else:
                    x1,y1,z11= torch.cat((x1,x)),  torch.cat((y1,y)), torch.cat((z11,z1.cpu()))
                count += 1
                if count % 100 ==0:
                    print('count: ', count)
    print('the size of samples:{}'.format(z11.shape))
    return z11.cpu()

def get_aug_z(augZ_file, pretrain_path, dataset = 'miniimagenet', weak = False, device = None):
    if not os.path.exists(augZ_file):
        print("encode aug_z")
        aug_z = encode_aug_z(pretrain_path, dataset = dataset, weak=weak, device= device)
        np.savez(augZ_file , Z = aug_z)
        print("encode finished,aug_z saved in: " + augZ_file)
    else:
        augZ = np.load(augZ_file)
        aug_z = augZ['Z']
        print("aug_z loaded from: " + augZ_file)
    return aug_z
        
def get_aug_matrix(samples, embs, cluter_center, augZ_file, augLabel_file, dataset, weak, KNN):
    if not os.path.exists(augLabel_file):
        print("trans by aug_z")
        clu_X_list,clu_Y_list,clu_pos_list,clu_emb_list, label_list = get_clustered_list(samples, embs)
        aug_z = get_aug_z(augZ_file, dataset = dataset, weak=weak)
        clu_aug_label_matrix, clu_aug_label_list, clu_stability_list, stable, stable_ratio, aug_label = get_aug_stable(embs, aug_z, cluter_center, clu_pos_list, KNN)
        np.savez(augLabel_file , aug_label_matrix = clu_aug_label_matrix, aug_label = aug_label)
        print("trans finished,aug_label saved in: " + augLabel_file)
    else:
        augLabel = np.load(augLabel_file)
        clu_aug_label_matrix = augLabel['aug_label_matrix']
        aug_label = augLabel['aug_label']
        print("aug_label loaded from: " + augLabel_file)
    return clu_aug_label_matrix, aug_label

def update_clu_centers(samples, clu_pos_list, clu_emb_list, aug_z, aug_label):
    Pos = samples["N"]
    X = samples["X"]
    Y = samples["Y"]
    clu_X_list_aug,clu_Y_list_aug,clu_pos_list_aug,clu_emb_list_aug, label_list_aug = get_clustered_list_1(Pos, X, Y, aug_label, aug_z)
    clu_centers_aug = []
    clu_centers_stable = []
    clu_centers_new = []
    for pos_list, pos_list_aug, emb_list, emb_list_aug in zip(clu_pos_list, clu_pos_list_aug, clu_emb_list, clu_emb_list_aug):
        num_aug = pos_list_aug.size
        pos_set = set(pos_list.tolist())
        pos_aug_set = set(pos_list_aug.tolist())
        stable_set = pos_set & pos_aug_set
        center_aug = np.asarray(emb_list_aug.mean(0))
        # 数量为0的问题没解决！！！！
        
        num_stable = len(stable_set)
        emb_stable = []
        emb_dict = dict(zip(pos_list, emb_list))
        for pos in stable_set:
            emb_stable.append(emb_dict[pos])
        emb_stable = np.asarray(emb_stable)
        center_stable = emb_stable.mean(0)
        center_new = (num_aug*center_aug+num_stable*center_stable)/(num_aug+num_stable)
        
        clu_centers_aug.append(center_aug)
        clu_centers_stable.append(center_stable)
        clu_centers_new.append(center_new)
    return clu_centers_stable, clu_centers_aug, clu_centers_new

def aug_reduce(pos, raw_emb, label, aug_emb, center, clu_pos_list, threshold = 0.3):
    clu_num = len(clu_pos_list)
    aug_emb_supply = aug_emb[pos]
    
    # KNN by clu centers
    estimator = KNeighborsClassifier(n_neighbors=1)
    center_label = list(range(clu_num))
    estimator.fit(center, center_label)
    aug_label = estimator.predict(aug_emb_supply)

    stable = label == aug_label
    stable_ratio = sum(stable)/len(label) 
    clu_stab_all_list = []
    clu_stab_ratio_list = []
    min_clu_no = 0
    min_value = 1
    for i, clu_pos in enumerate(clu_pos_list):
        clu_stab = stable[list(clu_pos)]
        stab_num = sum(clu_stab)
        clu_len = len(clu_pos)
        stab_ratio = stab_num/clu_len
        if stab_ratio < min_value:
            min_clu_no = i
            min_value = stab_ratio
        clu_stab_all_list.append((stab_ratio, stab_num, clu_len))
        clu_stab_ratio_list.append(stab_ratio)
        
    # store_hist_pic(clu_stab_ratio_list)  # plot stable_ratio statistics
    
    # reduce cluster with low aug_stable ratio
    print("start aug_reduce")
    reduced_label = np.copy(aug_label)
    reduced_clu = []
    reduced_pos = set()
    count = 0
    while min_value < threshold:
        count += 1
        reduced_clu.append(min_clu_no)
        center_label.remove(min_clu_no)
        min_clu_no_1 = min_clu_no
        for clu_no in reduced_clu:
            if clu_no < min_clu_no:
                min_clu_no_1 -= 1
        center = np.delete(center, min_clu_no_1, axis=0)
        estimator.fit(center, center_label)

        clu_pos = clu_pos_list[min_clu_no]
        # stable sample change label
        # clu_label = label[list(clu_pos)]
        # clu_aug_label = aug_label[list(clu_pos)]
        # stable_pos = clu_pos[np.nonzero(clu_label == clu_aug_label)]
        # if stable_pos.size > 0:
        #     aug_emb_stable = aug_emb_supply[stable_pos]
        #     stable_label_new = estimator.predict(aug_emb_stable)
        #     reduced_label[list(stable_pos)] = stable_label_new
        #     label_new = np.append(label_new, stable_label_new)
        
        # unstable sample change label
        # unstable_pos = clu_pos[np.nonzero(clu_label != clu_aug_label)]
        # if unstable_pos.size > 0:
        #     unstable_label_new = aug_label[list(unstable_pos)]
        #     reduced_label[list(unstable_pos)] = unstable_label_new
        #     label_new = np.append(label_new, unstable_label_new)
        
        # in sample change label
        # in_pos = np.where(aug_label == min_clu_no)[0]
        # if in_pos.size > 0:
        #     aug_emb_in = aug_emb_supply[in_pos]
        #     in_label_new = estimator.predict(aug_emb_in)
        #     reduced_label[list(in_pos)] = in_label_new
        #     label_new = np.append(label_new, in_label_new)
        
        # sample to be reduced
        pos_reducing = np.where(reduced_label == min_clu_no)[0]
        # pre_reduced_pos = set(pre_reduced_pos) - set(clu_pos)
        pos_reducing = list(pos_reducing)
        if len(pos_reducing) > 0:
            reduced_pos = reduced_pos | set(pos_reducing)
            aug_emb_red = aug_emb_supply[pos_reducing]
            label_new = estimator.predict(aug_emb_red)
            reduced_label[pos_reducing] = label_new
            
            # update aug_stable_ratio
            for l, pos in zip(label_new, pos_reducing):
                stab_ratio, stab_num, clu_len = clu_stab_all_list[l]
                # stab_num += 1
                # # reduce后trans的点可能就来自clu，数量需要去重
                # clu_pos0 = clu_pos_list[l]
                # if pos not in clu_pos0:
                #     clu_len += 1
                # stab_ratio = stab_num/clu_len
                # clu_stab_all_list[l] = (stab_ratio, stab_num, clu_len)
                # clu_stab_ratio_list[l] = stab_ratio
                
                # 只有trans回原类的点进行aug_stable更新 
                clu_pos0 = clu_pos_list[l]
                if pos in clu_pos0:
                    stab_num += 1
                    stab_ratio = stab_num/clu_len
                    clu_stab_all_list[l] = (stab_ratio, stab_num, clu_len)
                    clu_stab_ratio_list[l] = stab_ratio
            
        # update min
        clu_stab_all_list[min_clu_no] = (2, 0, 0)
        clu_stab_ratio_list[min_clu_no] = 2
        min_value = min(clu_stab_ratio_list)
        min_clu_no = clu_stab_ratio_list.index(min_value)
    print("end aug_reduce, reduced number:" + str(count))  
    return  reduced_label, center_label, center, reduced_clu, reduced_pos

def get_aug_reduced_label(samples, embs, clu_center, augZ_file, pretrain_path, reduced_file, dataset, weak, threshold, device):
    # print(reduced_file)
    if not os.path.exists(reduced_file):
        print("aug reduce start")
        pos = embs['N']
        raw_emb = embs['Z']
        label = embs['cluster_label']
        center = clu_center['cluster_centers']
        aug_z = get_aug_z(augZ_file, pretrain_path, dataset = dataset, weak=weak, device=device)
        # clu_X_list,clu_Y_list,clu_pos_list,clu_emb_list, label_list = get_clustered_list(samples, embs)
        clu_pos_list = get_clu_pos_list(samples)
        aug_label, stable_clu, stable_center, reduced_clu, reduced_pos = aug_reduce(pos, raw_emb, label, aug_z, center, clu_pos_list, threshold)
        np.savez(reduced_file , aug_label = aug_label, stable_clu=stable_clu, stable_center=stable_center, reduced_clu=reduced_clu, reduced_pos=reduced_pos)
        print("aug reduce finished, saved in: " + reduced_file)
    else:
        aug_reduce_file = np.load(reduced_file, allow_pickle=True)
        aug_label = aug_reduce_file['aug_label']
        stable_clu = aug_reduce_file['stable_clu']
        stable_center = aug_reduce_file['stable_center']
        reduced_clu = aug_reduce_file['reduced_clu']
        reduced_pos = aug_reduce_file['reduced_pos']
        print("aug_reduce loaded from: " + reduced_file)
    return aug_label, stable_clu, stable_center, reduced_clu, reduced_pos

if __name__ == '__main__':
    data_address = "./data/cfe_encodings/supply"
    samples_file = "miniimagenet_128_K_500_train_1698211778.npz"
    clu_center_file = "miniimagenet_128_K_500_train_clusters_1698211778.npz"
    emb_file = "miniimagenet_128_K_500_train_emb_1698211778.npz"
    no = samples_file.split('_')[-1][:-4]
    dataset = 'miniimagenet'
    
    sample_address = os.path.join(data_address, samples_file)
    emb_address = os.path.join(data_address, emb_file)
    samples = np.load(sample_address)
    embs = np.load(emb_address)
    
    clu_center_address = os.path.join(data_address, clu_center_file)
    clu_center = np.load(clu_center_address)
    K = 0
    clu_X_list,clu_Y_list,clu_pos_list,clu_emb_list, label_list = get_clustered_list(samples, embs)
    pretrain_path = "./pretrain/imagenet-embedding.pth.tar"
    augZ_file = './data/cfe_encodings/supply/{}_{}_augZ_weak{}.npz'.format(dataset, 128, "_1698211778")
    aug_z = get_aug_z(augZ_file, pretrain_path, dataset = dataset, weak=True)
    clu_aug_label_matrix, clu_aug_label_list, clu_stability_list, stable, stable_ratio,aug_emb_supply, aug_label = get_aug_stable(embs, aug_z, clu_center, clu_pos_list,K=K)
    # 函数有问题，数量为0的问题没有解决
    # clu_centers_aug = update_clu_centers(samples, clu_pos_list, clu_emb_list, aug_z, aug_label)
    
    label_dis_entro_dict = get_label_dis_entro(clu_Y_list)
    clu_entropy_list, clu_num_list, clu_purity_list, clu_mainL_dis_entro_list,clu_dis_entro_list = get_clu_entropy(clu_Y_list, label_dis_entro_dict)
    
    clu_stability_array = np.asarray(clu_stability_list)
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,0],y=clu_entropy_list,ci=50)
    # plt.xlabel("Semantic Stability")
    # plt.ylabel("Entropy")
    # plt.savefig('./plot/1698211778_/K{}_stab_entro.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,0],y=clu_num_list)
    # plt.savefig('./plot/1698211778_/K{}_stab_num.png'.format(K))
    plt.clf()
    sns.regplot(x=clu_stability_array[:,0], y=clu_purity_list, 
                fit_reg=False, scatter_kws={"s":clu_num_list} )
    plt.xlabel("Semantic Stability")
    plt.ylabel("Purity")
    plt.savefig('./plot/1698211778_/K{}_stab_pure.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,0], y=clu_mainL_dis_entro_list)
    # plt.savefig('./plot/1698211778_/K{}_stab_mainDis.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,0], y=clu_dis_entro_list)
    # plt.savefig('./plot/1698211778_/K{}_stab_dis.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,0], y=np.asarray(clu_entropy_list)/clu_dis_entro_list)
    # plt.savefig('./plot/1698211778_/K{}_stab_dis1.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,1], y=clu_entropy_list)
    # plt.savefig('./plot/1698211778_/K{}_stabN_entro.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,2], y=clu_entropy_list,)
    # plt.savefig('./plot/1698211778_/K{}_size_entro.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,2], y=clu_stability_array[:,0], )
    # plt.savefig('./plot/1698211778_/K{}_size_stab.png'.format(K))
    # plt.clf()
    # sns.regplot(x=clu_stability_array[:,0]*clu_stability_array[:,2],y=clu_entropy_list)
    # plt.xlabel("stable_ratio")
    # plt.ylabel("normalized_entropy")
    # plt.savefig('./plot/1698211778_/K{}_normStab_entro.png'.format(K))
    
    
    
