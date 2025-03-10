import os
import torch
import torch.nn.functional as F
import math
import time
import json
import logging
import statistics
import numpy as np
import time

from torchmeta.utils.data import BatchMetaDataLoader
from datasets_cluster import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.metalearners.maml_pem_aug import ModelAgnosticMetaLearningAug

from shutil import copyfile
from utils import momentum_update, get_last_checkpoint
from aug_stable.aug_stable import get_aug_stable, get_aug_z, get_aug_matrix, aug_reduce, get_aug_reduced_label
from plot import get_clustered_list

def main(args):
    train_un_maml_pem_aug(args)

# aug语义一致性
def train_un_maml_pem_aug(args):
    resume = False
    save_delta = 100
    if args.resume_config is not None:
        print('{}:resume from `{}`'.format(time.asctime(time.localtime(time.time())),os.path.abspath(args.resume_config)))
        args_dict = vars(args)
        with open(args.resume_config, 'rt') as f:
            args_pre = json.load(f)
            args_pre['cuda_no'] = args.cuda_no
            args_pre['num_epochs'] = args.num_epochs
            args_dict.update(args_pre)
        resume = True
    elif args.output_folder is not None:
        args.model_path = os.path.abspath(os.path.join(args.output_folder, 'model.th'))
        # Save the configuration in a config.json file
        args.config = os.path.join(args.output_folder, 'config.json')
        with open(args.config, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print('Saving configuration file in `{0}`'.format(
                     os.path.abspath(args.config)))
        
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda:{}'.format(args.cuda_no) if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')
    print('device:{}'.format(device))
    log_file = os.path.join(args.output_folder, "log.txt")

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shots,
                                      num_test_per_class=args.num_shots_test)
    # class_augmentations = [Rotation([90, 180, 270])]
    class_augmentations = None

    if args.dataset == 'omniglot':
        encoding_dim = 64
        save_delta = 10
        t_dict = {500:'_1704817310'}
        t = t_dict[int(args.K)]
        filename_data = 'omniglot_64_K_{}_%s{}.npz'.format(args.K, t)
        filename_clusters='omniglot_64_K_{}_%s_clusters{}.npz'.format(args.K, t)
        filename_emb='omniglot_64_K_{}_%s_emb{}.npz'.format(args.K, t)

        folder=os.path.join('cfe_encodings',args.cluster_supply)
        sample_address = os.path.join(args.folder, folder, filename_data)
        emb_address = os.path.join(args.folder, folder, filename_emb)
        cluster_address = os.path.join(args.folder, folder, filename_clusters)
        samples = np.load(sample_address % "train")
        embs = np.load(emb_address % "train")
        cluter_center = np.load(cluster_address % "train") 
        
        weak = 2
        KNN = 0 
        threshold = 0.7
        fix_raw = True
        if weak >= 0:
            augZ_file = './data/cfe_encodings/supply/{}_{}_augZ_K{}_weak{}.npz'.format(args.dataset, encoding_dim, args.K, weak)
            augLabel_file = './data/cfe_encodings/supply/{}_{}_augLabel_KNN{}_K{}_weak{}.npz'.format(args.dataset, encoding_dim, KNN, args.K, weak)
            if fix_raw:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_fix_Thr{}_K{}_weak{}.npz'.format(args.dataset, encoding_dim, threshold, args.K, weak)
            else:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_Thr{}_K{}_weak{}.npz'.format(args.dataset, encoding_dim, threshold, args.K, weak)
        else:
            augZ_file = './data/cfe_encodings/supply/{}_{}_augZ_K{}.npz'.format(args.dataset, encoding_dim, args.K)
            augLabel_file = './data/cfe_encodings/supply/{}_{}_augLabel_KNN{}_K{}.npz'.format(args.dataset, encoding_dim, KNN, args.K)
            if fix_raw:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_fix_Thr{}_K{}.npz'.format(args.dataset, encoding_dim, threshold, args.K)
            else:
                augReduce_file = './data/cfe_encodings/supply/{}_{}_augReduce_Thr{}_K{}.npz'.format(args.dataset, encoding_dim, threshold, args.K)
        clu_aug_label_matrix = None
        pretrain_path = "./pretrain/omni-embedding.pth.tar"

        aug_label, stable_clu, stable_center, reduced_clu, reduced_pos = get_aug_reduced_label(samples, embs, cluter_center, augZ_file, pretrain_path, augReduce_file, args.dataset, weak, threshold,device)
        # aug_label = None  
        # aug_label = samples['Y']  

        transform = None
        meta_train_dataset = Omniglot(args.folder, transform=transform,
            target_transform=Categorical(args.num_ways),
            num_classes_per_task=args.num_ways, meta_train=True,
            class_augmentations=class_augmentations,
            dataset_transform=dataset_transform, download=True,
            folder=folder, filename= filename_data, aug_label=aug_label)
        
        meta_val_dataset = Omniglot(args.folder, transform=transform,
            target_transform=Categorical(args.num_ways),
            num_classes_per_task=args.num_ways, meta_val=True,
            class_augmentations=class_augmentations,
            dataset_transform=dataset_transform,
            folder=folder)

        model = ModelConvOmniglot(args.num_ways, hidden_size=args.hidden_size)
        loss_function = F.cross_entropy

    elif args.dataset == 'miniimagenet' or args.dataset == 'tieredimagenet':
        encoding_dim = 128
        if args.dataset == 'miniimagenet':
            t = '_1698211778'
            filename_data='miniimagenet_128_K_{}_%s{}.npz'.format(args.K, t)
            filename_clusters='miniimagenet_128_K_{}_%s_clusters{}.npz'.format(args.K, t)
            filename_emb='miniimagenet_128_K_{}_%s_emb{}.npz'.format(args.K, t)
        else:
            t = '_1705373773'
            filename_data='tieredimagenet_128_K_{}_%s{}.npz'.format(args.K, t)
            filename_clusters='tieredimagenet_128_K_{}_%s_clusters{}.npz'.format(args.K, t)
            filename_emb='tieredimagenet_128_K_{}_%s_emb{}.npz'.format(args.K, t)
        
        folder=os.path.join('cfe_encodings',args.cluster_supply)
        sample_address = os.path.join(args.folder, folder, filename_data)
        emb_address = os.path.join(args.folder, folder, filename_emb)
        cluster_address = os.path.join(args.folder, folder, filename_clusters)
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
        clu_aug_label_matrix = None
        pretrain_path = "./pretrain/imagenet-embedding.pth.tar"
        aug_label, stable_clu, stable_center, reduced_clu, reduced_pos = get_aug_reduced_label(samples, embs, cluter_center, augZ_file, pretrain_path, augReduce_file, args.dataset, weak, threshold,device)
        # aug_label = None 
        # aug_label = samples['Y']
        
        transform = None
        if args.dataset == 'miniimagenet':
            meta_train_dataset = MiniImagenet(args.folder, transform=transform,
                                              target_transform=Categorical(args.num_ways),
                                              num_classes_per_task=args.num_ways, meta_train=True,
                                              class_augmentations=class_augmentations,
                                              dataset_transform=dataset_transform, download=True, 
                                              folder=folder ,filename= filename_data, aug_label=aug_label)
            meta_val_dataset = MiniImagenet(args.folder, transform=transform,
                                            target_transform=Categorical(args.num_ways),
                                            num_classes_per_task=args.num_ways, meta_val=True,
                                            class_augmentations=class_augmentations,
                                            dataset_transform=dataset_transform,
                                            folder=folder)
        else:
            meta_train_dataset = TieredImagenet(args.folder, transform=transform,
                                                target_transform=Categorical(args.num_ways),
                                                num_classes_per_task=args.num_ways, meta_train=True,
                                                class_augmentations=class_augmentations,
                                                dataset_transform=dataset_transform, download=True,
                                                folder=folder ,filename= filename_data, aug_label=aug_label)
            meta_val_dataset = TieredImagenet(args.folder, transform=transform,
                                              target_transform=Categorical(args.num_ways),
                                              num_classes_per_task=args.num_ways, meta_val=True,
                                              class_augmentations=class_augmentations,
                                              dataset_transform=dataset_transform,
                                              folder=folder)
        model = ModelConvMiniImagenet(args.num_ways, hidden_size=args.hidden_size)  
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(args.dataset))

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True)

    best_value = None
    fail_count = 0
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    epoch_start = -1
    if resume:
        if args.checkpoint is None:
            args.checkpoint = get_last_checkpoint(args.output_folder)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.to(device=device)
        model.load_state_dict(checkpoint['parameter'])
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
        meta_optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        best_value = checkpoint['best_value']
    else:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr) 
    metalearner = ModelAgnosticMetaLearningAug(model, meta_optimizer, dataloader=meta_train_dataloader,
        first_order=args.first_order, num_adaptation_steps=args.num_steps,
        step_size=args.step_size, loss_function=loss_function, device=device, 
        clu_aug_label_matrix=clu_aug_label_matrix)

    # Training loop
    start_time = time.time()
    for epoch in range(epoch_start+1, args.num_epochs):
        if epoch<args.n_warmup or aug_label is not None:
            pem_eta = 1
        else:
            pem_eta = 0.9
        # print('progress evaluation eta={}'.format(pem_eta))
        metalearner.train(meta_train_dataloader, log_file, max_batches=args.num_batches, pem_eta=pem_eta,
                          verbose=args.verbose, desc='Training', leave=False)
        
        momentum_update(metalearner.model, metalearner.eval_model,
                        replace=True)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))

        # Save best model
        if (best_value is None) or (('accuracies_after' in results)
                and (best_value < results['accuracies_after'])):
            best_value = results['accuracies_after']
            save_model = True
        # elif (best_value is None) or (best_value > results['mean_outer_loss']):
        #     best_value = results['mean_outer_loss']
        #     save_model = True
        else:
            save_model = False
            
        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(model.state_dict(), f)

        if (epoch+1)%save_delta==0:
            copyfile(args.model_path, '{}_best_epoch_{}.th'.format(args.model_path.split('.th')[0], epoch+1))
            checkpoint = {'parameter': model.state_dict(),
                          'optimizer': meta_optimizer.state_dict(),
                          'epoch': epoch,
                          'best_value': best_value}
            torch.save(checkpoint, '{}_epoch_{}.th'.format(args.model_path.split('.th')[0],epoch+1))
        if save_model:
            fail_count = 0
        else:
            fail_count+=1
            # print('fail counts = {}'.format(fail_count))
            # if fail_count >= 2:
            #     # init_fr *= fr_momentum
            #     fail_count = 0
        print('epoch = {}, best value = {:.4f}, current value = {:.4f}, fail counts = {}'.format(epoch, best_value, results['accuracies_after'], fail_count))
        # with open(log_file, 'a') as f:
        #     f.write('epoch = {}, best value = {:.4f}, current value = {:.4f}, fail counts = {}\n'.format(epoch, best_value, results['accuracies_after'], fail_count))
        log_time = time.time() 
        elapsed_time_log = log_time - start_time
        minutes_log = int(elapsed_time_log // 60)
        seconds_log = int(elapsed_time_log % 60)
        print(f"consumed time: {minutes_log} minutes {seconds_log} seconds")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training time: {minutes} minutes {seconds} seconds")

    if hasattr(meta_train_dataset, 'close'):
        meta_train_dataset.close()
        meta_val_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str, 
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet','tieredimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default='./tmp',
        help='Path to the output folder to save the model.')
    parser.add_argument('--config', type=str, default=None,
        help='Path to the output folder to save the config.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=5,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')
    parser.add_argument('--K', type=int, default=500,
        help='Number of clusters')
    parser.add_argument('--resume_config', type=str, default=None,
        help='config path for resume.')
    parser.add_argument('--checkpoint', type=str, default=None,
        help='checkpoint uesd to resume.')
    parser.add_argument('--cluster-supply', type=str, default='supply',
                        help='supply cluster num.')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=8,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--cuda-no', type=int, default=0,
        help='Number of cuda')
    # Progress Evaluation
    # parser.add_argument('--eval-beta', type=float, default=0.9,
    #                     help='beta is a parameter to control the momentum updating for the eval model'
    #                          '(default: 0.9).')
    parser.add_argument('--n-warmup', type=int, default=10,
                        help='the number of warm up')
    
    # method control
    parser.add_argument('--supply', type=int, default=0,
                        help='method to supply')
    parser.add_argument('--new_method', type=int, default=0,
                        help='method to get new samples')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
