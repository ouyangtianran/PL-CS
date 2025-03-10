import os
# os.environ['CUDA_VISIBLE_DEVICE']='1'

import torch
import argparse
import pandas as pd
import sys
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import utils as ut
import torchvision
import numpy as np
import time

from src import datasets, models
from src.models import backbones
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler

from aug_stable.aug_stable import get_aug_reduced_label

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


def trainval(exp_dict, savedir_base, datadir, reset=False, 
            num_workers=0, pretrained_weights_dir=None):

    # get experiment directory
    # exp_id = hu.hash_dict(exp_dict)
    exp_id = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)  

    save_delta = 100
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda_no))
        print(device)
    else:
        print("No GPU is available.")
        
    if args.dataset == "omniglot":
        encoding_dim = 64
        save_delta = 10
        t_dict = {500:'_1704817310', 2000:'_1704974311', 3000:'_1704974573'}
        t = t_dict[int(args.K)]
        filename_data_valtest = 'omniglot_64_K_{}_%s.npz'.format(args.K)
        filename_data = 'omniglot_64_K_{}_%s{}.npz'.format(args.K, t)
        filename_clusters='omniglot_64_K_{}_%s_clusters{}.npz'.format(args.K, t)
        filename_emb='omniglot_64_K_{}_%s_emb{}.npz'.format(args.K, t)

        folder=os.path.join('cfe_encodings',args.cluster_supply)
        sample_address = os.path.join(args.datadir, folder, filename_data)
        emb_address = os.path.join(args.datadir, folder, filename_emb)
        cluster_address = os.path.join(args.datadir, folder, filename_clusters)
        samples = np.load(sample_address % "train")
        embs = np.load(emb_address % "train")
        cluter_center = np.load(cluster_address % "train") 
        
        weak = 2
        KNN = 0 
        threshold = 0.7
        fix_raw = True
        if weak >= 0:
            augZ_file = './data/cfe_encodings/supply/{}_{}_augZ_K{}_weak{}{}.npz'.format(args.dataset, encoding_dim, args.K, weak ,t)
            augLabel_file = './data/cfe_encodings/supply/{}_{}_augLabel_KNN{}_K{}_weak{}{}.npz'.format(args.dataset, encoding_dim, KNN, args.K, weak ,t)
            if fix_raw:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_fix_Thr{}_K{}_weak{}{}.npz'.format(args.dataset, encoding_dim, threshold, args.K, weak ,t)
            else:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_Thr{}_K{}_weak{}{}.npz'.format(args.dataset, encoding_dim, threshold, args.K, weak ,t)
        else:
            augZ_file = './data/cfe_encodings/supply/{}_{}_augZ_K{}{}.npz'.format(args.dataset, encoding_dim, args.K ,t)
            augLabel_file = './data/cfe_encodings/supply/{}_{}_augLabel_KNN{}_K{}{}.npz'.format(args.dataset, encoding_dim, KNN, args.K ,t)
            if fix_raw:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_fix_Thr{}_K{}{}.npz'.format(args.dataset, encoding_dim, threshold, args.K ,t)
            else:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_Thr{}_K{}{}.npz'.format(args.dataset, encoding_dim, threshold, args.K ,t)
        aug_label, stable_clu, stable_center, reduced_clu, reduced_pos = get_aug_reduced_label(samples, embs, cluter_center, augZ_file, augReduce_file, args.dataset, weak, threshold,device)
        # aug_label = None  
        # aug_label = samples['Y']  
        del(samples, embs, cluter_center)
    
    else:
        encoding_dim = 128
        if args.dataset == 'miniimagenet':
            t = '_1698211778'
            filename_data_valtest = 'miniimagenet_128_K_{}_%s.npz'.format(args.K)
            filename_data='miniimagenet_128_K_{}_%s{}.npz'.format(args.K, t)
            filename_clusters='miniimagenet_128_K_{}_%s_clusters{}.npz'.format(args.K, t)
            filename_emb='miniimagenet_128_K_{}_%s_emb{}.npz'.format(args.K, t)
        else:
            # t = '_1705373773'
            t = '_1709806219'
            if args.K == 1000:
                t = '_1705375232'
            filename_data_valtest = 'tieredimagenet_128_K_500_%s.npz' # valtest dataset is same between K500 and K1000
            filename_data='tieredimagenet_128_K_{}_%s{}.npz'.format(args.K, t)
            filename_clusters='tieredimagenet_128_K_{}_%s_clusters{}.npz'.format(args.K, t)
            filename_emb='tieredimagenet_128_K_{}_%s_emb{}.npz'.format(args.K, t)
        
        folder=os.path.join('cfe_encodings',args.cluster_supply)
        sample_address = os.path.join(args.datadir, folder, filename_data)
        emb_address = os.path.join(args.datadir, folder, filename_emb)
        cluster_address = os.path.join(args.datadir, folder, filename_clusters)
        samples = np.load(sample_address % "train")
        embs = np.load(emb_address % "train")
        cluter_center = np.load(cluster_address % "train")
        
        weak = True
        KNN = 0 
        threshold = 0.7
        fix_raw = True
        if weak:
            augZ_file = './data/cfe_encodings/supply/{}_{}_augZ_weak{}.npz'.format(args.dataset, encoding_dim, t)
            augLabel_file = './data/cfe_encodings/supply/{}_{}_augLabel_KNN{}_weak{}.npz'.format(args.dataset, encoding_dim, KNN, t)
            if fix_raw:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_fix_Thr{}_weak{}.npz'.format(args.dataset, encoding_dim, threshold, t)
            else:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_Thr{}_weak{}.npz'.format(args.dataset, encoding_dim, threshold, t)
        else:
            augZ_file = './data/cfe_encodings/supply/{}_{}_augZ{}.npz'.format(args.dataset, encoding_dim, t)
            augLabel_file = './data/cfe_encodings/supply/{}_{}_augLabel_KNN{}{}.npz'.format(args.dataset, encoding_dim, KNN, t)
            if fix_raw:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_fix_Thr{}{}.npz'.format(args.dataset, encoding_dim, threshold, t)
            else:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_Thr{}{}.npz'.format(args.dataset, encoding_dim, threshold, t)
        aug_label, stable_clu, stable_center, reduced_clu, reduced_pos = get_aug_reduced_label(samples, embs, cluter_center, augZ_file, augReduce_file, args.dataset, weak, threshold,device)
        # aug_label = None  
        # aug_label = samples['Y']  
    
        # 0-1 => 0-255
        raw_filename_data = "raw_" + filename_data
        raw_filename_data_valtest = "raw_" + filename_data_valtest
        raw_addr = os.path.join(args.datadir, folder, raw_filename_data)
        addr_valtest = os.path.join(args.datadir, folder, filename_data_valtest)
        raw_addr_valtest = os.path.join(args.datadir, folder, raw_filename_data_valtest)
        if not os.path.exists(raw_addr % "train"):
            print("train data: 0-1 => 0-255")
            np.savez(raw_addr % "train" ,N = samples["N"], X = (samples["X"]*255).astype(np.uint8), Y=samples["Y"], cluster_label=samples["cluster_label"])
            del(samples,embs,cluter_center)
        if not os.path.exists(raw_addr_valtest % "val"):
            print("val data: 0-1 => 0-255")
            samples_val = np.load(addr_valtest % "val")
            np.savez(raw_addr_valtest % "val" ,X = (samples_val["X"]*255).astype(np.uint8), Y=samples_val["Y"], cluster_label=samples_val["cluster_label"])
            del(samples_val)
        if not os.path.exists(raw_addr_valtest % "test"):
            print("test data: 0-1 => 0-255")
            samples_test = np.load(addr_valtest % "test")
            np.savez(raw_addr_valtest % "test" ,X = (samples_test["X"]*255).astype(np.uint8), Y=samples_test["Y"], cluster_label=samples_test["cluster_label"])
            del(samples_test)
        filename_data = raw_filename_data
        filename_data_valtest = raw_filename_data_valtest

    # shrink aug_label
    exp_dict["n_classes"] = len(stable_clu)
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)  
    aug_label_1 = ut.shrink_label(aug_label, stable_clu)

    # load datasets
    # ==========================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset_train"],
                data_root=os.path.join(datadir, folder),
                split="train", 
                transform=exp_dict["transform_train"], 
                classes=exp_dict["classes_train"],
                support_size=exp_dict["support_size_train"],
                query_size=exp_dict["query_size_train"], 
                n_iters=exp_dict["train_iters"],
                unlabeled_size=exp_dict["unlabeled_size_train"],
                filename = filename_data, aug_label=aug_label_1)

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset_val"],
                data_root=os.path.join(datadir, folder),
                split="val", 
                transform=exp_dict["transform_val"], 
                classes=exp_dict["classes_val"],
                support_size=exp_dict["support_size_val"],
                query_size=exp_dict["query_size_val"], 
                n_iters=exp_dict.get("val_iters", None),
                unlabeled_size=exp_dict["unlabeled_size_val"],
                filename = filename_data_valtest)

    test_set = datasets.get_dataset(dataset_name=exp_dict["dataset_test"],
                data_root=os.path.join(datadir, folder),
                split="test", 
                transform=exp_dict["transform_val"], 
                classes=exp_dict["classes_test"],
                support_size=exp_dict["support_size_test"],
                query_size=exp_dict["query_size_test"], 
                n_iters=exp_dict["test_iters"],
                unlabeled_size=exp_dict["unlabeled_size_test"],
                filename = filename_data_valtest)

    # get dataloaders
    # ==========================
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate(exp_dict["collate_fn"]),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)

    
    # create model and trainer
    # ==========================

    # Create model, opt, wrapper
    backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone, 
                                 n_classes=exp_dict["n_classes"],
                                 exp_dict=exp_dict,
                                 pretrained_weights_dir=pretrained_weights_dir,
                                 savedir_base=savedir_base)
    
    # Pretrain or Fine-tune or run SSL
    if exp_dict["model"]['name'] == 'ssl':
        # runs the SSL experiments
        score_list_path = os.path.join(savedir, 'score_list.pkl')
        if not os.path.exists(score_list_path):
            test_dict = model.test_on_loader(test_loader, max_iter=None)
            hu.save_pkl(score_list_path, [test_dict])
        return 
        
    # Checkpoint
    # -----------
    checkpoint_path = os.path.join(savedir, 'checkpoint.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(checkpoint_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}
        score_dict.update(model.get_lr())
        
        # train
        score_dict.update(model.train_on_loader(train_loader))

        # validate
        score_dict.update(model.val_on_loader(val_loader))
        # score_dict.update(model.test_on_loader(test_loader))

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())

        # Save checkpoint
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(checkpoint_path, model.get_state_dict())
        print("Saved: %s" % savedir)

        if "accuracy" in exp_dict["target_loss"]:
            is_best = score_dict[exp_dict["target_loss"]] >= score_df[exp_dict["target_loss"]][:-1].max() 
        else:
            is_best = score_dict[exp_dict["target_loss"]] <= score_df[exp_dict["target_loss"]][:-1].min() 

        # Save best checkpoint
        if is_best:
            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "checkpoint_best.pth"), model.get_state_dict())
            print("Saved Best: %s" % savedir)  
        
        # Check for end of training conditions
        if model.is_end_of_training():
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default='')
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', type=str, default=None)
    parser.add_argument('-j', '--run_jobs', type=int, default=0)
    parser.add_argument('-nw', '--num_workers', default=0, type=int)
    parser.add_argument('-p', '--pretrained_weights_dir', type=str, default=None)
    
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet','tieredimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--K', type=int, default=500,
                        help='Number of clusters')
    parser.add_argument('--resume_config', type=str, default=None,
                        help='config path for resume.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint uesd to resume.')
    parser.add_argument('--cluster-supply', type=str, default='supply',
                        help='supply cluster num.')
    # parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--cuda-no', type=int, default=0,
        help='Number of cuda')
    parser.add_argument('--aug-label', type=bool, default=False)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        pass
    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir=args.datadir,
                    reset=args.reset,
                    num_workers=args.num_workers,
                    pretrained_weights_dir=args.pretrained_weights_dir)