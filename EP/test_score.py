import json
import torch
import argparse
import os
import numpy as np

from src import datasets, models
from src.models import backbones
from torch.utils.data import DataLoader

def test(checkpoint_dir, datadir='../data', ways=5, shots=1,
         num_workers=4, filename_test=None):
    folder=os.path.join('cfe_encodings',args.cluster_supply)
    json_name = os.path.join(checkpoint_dir,'exp_dict.json')
    pretrained_weights_dir = checkpoint_dir
    savedir_base = os.path.join(checkpoint_dir,'results')
    if not os.path.exists(savedir_base):
        os.makedirs(savedir_base)
    # print(fold_name)

    with open(json_name) as f:
        exp_dict = json.load(f)
    exp_dict['ngpu'] = 1
    exp_dict["classes_test"] = ways
    exp_dict["support_size_test"] = shots
    test_set = datasets.get_dataset(dataset_name=exp_dict["dataset_test"],
                                    data_root=os.path.join(datadir, folder),
                                    split="test",
                                    transform=exp_dict["transform_val"],
                                    classes=exp_dict["classes_test"],
                                    support_size=exp_dict["support_size_test"],
                                    query_size=exp_dict["query_size_test"],
                                    n_iters=exp_dict["test_iters"],
                                    unlabeled_size=exp_dict["unlabeled_size_test"],
                                    filename=filename_test)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)
    # Create model, opt, wrapper
    backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone,
                             n_classes=exp_dict["n_classes"],
                             exp_dict=exp_dict,
                             pretrained_weights_dir=None,
                             checkpoint=checkpoint_dir,
                             savedir_base=checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    # checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    checkpoint = torch.load((checkpoint_path))
    model.load_state_dict(checkpoint)
    model.cuda()

    test_dict = model.test_on_loader(test_loader, max_iter=None)
    print("Test on: "+checkpoint_dir+", {} way {} shot:".format(ways, shots))
    # print('dataset_train_{} dataset_test_root {}'.format(exp_dict["dataset_train"], exp_dict["dataset_test_root"]))
    print(test_dict)
    with open(os.path.join(savedir_base, 'results_{}_ways_{}_shots_dataset_train_{}.json'.format(ways, shots, exp_dict["dataset_train"])), 'w') as f:
        json.dump(test_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--savedir_base', default='./logs/finetune-omni')
    parser.add_argument('-d', '--datadir', default='../data')
    parser.add_argument('--K', type=int, default=500, help='Number of clusters')
    parser.add_argument('--cluster-supply', type=str, default='supply', help='supply cluster num.')
    args = parser.parse_args()

    d = args.savedir_base #'./logs/finetune-omni'
    ll = os.listdir(d)
    ways_omni = [5,5,20,20]
    shots_omni = [1,5,1,5]
    ways_mini = [5,5,5,5]
    shots_mini = [1,5,20,50]
    # ll= ['794ecddfb5678e63abba1ca8ca10e6d8','3cbc261ea41499bacc7490ceb116f9c0']
    # ll=['f31d5ab40808295a9d28c95d0db93b91']
    for i, l in enumerate(ll):
        cd = os.path.join(d,l)
        results = os.path.join(cd, "results")
        if os.path.exists(results):
            continue
        json_name = os.path.join(cd, 'exp_dict.json')
        if os.path.exists(json_name):
            with open(json_name) as f:
                exp_dict = json.load(f)
        else:
            continue
        if not ('conv' in exp_dict['model']['backbone']):
            continue
        if 'omniglot' in exp_dict["dataset_train"]:
            ww, ss = ways_omni, shots_omni
            t_dict = {500:'_1704817310', 2000:'_1704974311', 3000:'_1704974573'}
            t = t_dict[int(args.K)]
            filename_data_valtest = 'omniglot_64_K_{}_%s.npz'.format(args.K)
        elif 'imagenet' in exp_dict["dataset_train"]:
            ww, ss = ways_mini, shots_mini
            if 'mini' in exp_dict["dataset_train"]:
                filename_data_valtest = 'miniimagenet_128_K_{}_%s.npz'.format(args.K)
            else:  # tiered
                filename_data_valtest = 'tieredimagenet_128_K_{}_%s.npz'.format(args.K)
            raw_filename_data_valtest = "raw_" + filename_data_valtest
            folder=os.path.join('cfe_encodings',args.cluster_supply)
            addr_valtest = os.path.join(args.datadir, folder, filename_data_valtest)
            raw_addr_valtest = os.path.join(args.datadir, folder, raw_filename_data_valtest)
            if not os.path.exists(raw_addr_valtest % "test"):
                print("test data: 0-1 => 0-255")
                samples_test = np.load(addr_valtest % "test")
                np.savez(raw_addr_valtest % "test" ,X = samples_test["X"]*255, Y=samples_test["Y"], cluster_label=samples_test["cluster_label"])
            filename_data_valtest = raw_filename_data_valtest
        for w, s in zip(ww,ss):
            test(cd,datadir=args.datadir, ways=w,shots=s,filename_test = filename_data_valtest)
