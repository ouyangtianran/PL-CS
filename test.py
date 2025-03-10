import torch
import torch.nn.functional as F
import os
import json

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.metalearners import ModelAgnosticMetaLearning
from maml.utils import ToTensor1D

def main(args):
    # test(args)
    test_all(args)
   
def test_all(args):
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        with open(os.path.join(args.folder, 'config.json') , 'r') as f:
            config = json.load(f)
    if args.folder is not None:
        config['folder'] = args.folder
    if args.test_num_steps > 0:
        config['num_steps'] = args.test_num_steps
    if args.test_num_batches > 0:
        config['num_batches'] = args.test_num_batches
    if args.test_num_shots > 0:
        config['num_shots'] = args.test_num_shots
    if args.model_path != '0':
        config['model_path']=args.model_path
        
    device = torch.device('cuda:{}'.format(args.cuda_no) if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(device)
    
    # get model list
    dirname = os.path.dirname(config['model_path'])
    file_list = os.listdir(dirname)
    model_dict = dict()
    for file in file_list:
        if file.startswith("model_epoch_"):
            name = file.split('.th')[0]
            epoch = name.split('_')[-1]
            model_dict[epoch] = (file, "model_best_epoch_{}.th".format(epoch))
            
    # get dataset dict
    num_ways = config['num_ways']
    if config['dataset'] == 'sinusoid':
        transform = ToTensor1D()
        meta_test_dataset = Sinusoid(config['num_shots'] + config['num_shots_test'],
            num_tasks=1000000, transform=transform, target_transform=transform,
            dataset_transform=dataset_transform)
        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif config['dataset'] == 'omniglot':
        num_shots_list = [1,5]
        dataset_transform_list = [ClassSplitter(shuffle=True, num_train_per_class=num_shots,num_test_per_class=config['num_shots_test']) 
                                for num_shots in num_shots_list]
        transform = Compose([Resize(28), ToTensor()])
        meta_test_dataset_dict = {}
        model = ModelConvOmniglot(num_ways, hidden_size=config['hidden_size'])
        for count, num_shots in enumerate(num_shots_list):
            meta_test_dataset = Omniglot(config['folder'], transform=transform,target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways, meta_test=True,
                                         dataset_transform=dataset_transform_list[count], download=True)
            meta_test_dataset_dict[(num_ways,num_shots)] = meta_test_dataset 
        loss_function = F.cross_entropy

    elif config['dataset'] == 'miniimagenet':
        # num_shots_list = [1,5]
        num_shots_list = [1,5, 20,50]
        dataset_transform_list = [ClassSplitter(shuffle=True, num_train_per_class=num_shots,num_test_per_class=config['num_shots_test']) 
                                for num_shots in num_shots_list]
        transform = Compose([Resize(84), ToTensor()])
        meta_test_dataset_dict = {}
        model = ModelConvMiniImagenet(num_ways, hidden_size=config['hidden_size'])
        for count, num_shots in enumerate(num_shots_list):
            meta_test_dataset = MiniImagenet(config['folder'], transform=transform, target_transform=Categorical(num_ways),
                                             num_classes_per_task=num_ways, meta_test=True,
                                             dataset_transform=dataset_transform_list[count], download=True)
            meta_test_dataset_dict[(num_ways,num_shots)] = meta_test_dataset       
        loss_function = F.cross_entropy
    elif config['dataset'] == 'tieredimagenet':
        # num_shots_list = [1,5]
        num_shots_list = [1,5,20,50]
        dataset_transform_list = [ClassSplitter(shuffle=True, num_train_per_class=num_shots,num_test_per_class=config['num_shots_test']) 
                                for num_shots in num_shots_list]
        transform = Compose([Resize(84), ToTensor()])
        meta_test_dataset_dict = {}
        model = ModelConvMiniImagenet(num_ways, hidden_size=config['hidden_size'])
        for count, num_shots in enumerate(num_shots_list):
            meta_test_dataset = TieredImagenet(config['folder'], transform=transform, target_transform=Categorical(num_ways),
                                             num_classes_per_task=num_ways, meta_test=True,
                                             dataset_transform=dataset_transform_list[count], download=True)
            meta_test_dataset_dict[(num_ways,num_shots)] = meta_test_dataset 
        loss_function = F.cross_entropy
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(config['dataset']))
    
    # get dataloader dict
    meta_test_dataloader_dict = {}
    for key in meta_test_dataset_dict.keys():
        meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset_dict[key], batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True)
        meta_test_dataloader_dict[key] = meta_test_dataloader
    
    # test all models
    for e in sorted(model_dict.keys()):
        if int(e) != 200: continue
        if int(e) > 400: break
        model_name, model_best_name = model_dict[e]
        if not args.best_only:
            checkpoint1 = torch.load(os.path.join(dirname, model_name), map_location=device)
            model.load_state_dict(checkpoint1["parameter"])
            metalearner1 = ModelAgnosticMetaLearning(model,first_order=config['first_order'], num_adaptation_steps=config['num_steps'],
                                                    step_size=config['step_size'], loss_function=loss_function, device=device)
        checkpoint2 = torch.load(os.path.join(dirname, model_best_name), map_location=device)
        model.load_state_dict(checkpoint2)
        metalearner2 = ModelAgnosticMetaLearning(model,first_order=config['first_order'], num_adaptation_steps=config['num_steps'],
                                                step_size=config['step_size'], loss_function=loss_function, device=device)
        for num_shots in num_shots_list:
            results2 = metalearner2.evaluate(meta_test_dataloader_dict[(num_ways,num_shots)], max_batches=config['num_batches'], 
                                        verbose=args.verbose, desc='Test')
            if not args.best_only:
                results1 = metalearner1.evaluate(meta_test_dataloader_dict[(num_ways,num_shots)], max_batches=config['num_batches'], 
                                            verbose=args.verbose, desc='Test')
            
                print('{}ways_{}shots_epoch{}:ACC {:.4f} {:.4f},LOSS {:.4f} {:.4f}'.format(num_ways, num_shots, e,
                                                                                                results1['accuracies_after'], results2['accuracies_after'],
                                                                                                results1['mean_outer_loss'], results2['mean_outer_loss']))
                with open(os.path.join(dirname, 'log_results.txt'), 'a+') as f:
                    f.write('{}ways_{}shots_epoch{}:ACC {:.4f} {:.4f},LOSS {:.4f} {:.4f}\n'.format(num_ways, num_shots, e,
                                                                                                results1['accuracies_after'], results2['accuracies_after'],
                                                                                                results1['mean_outer_loss'], results2['mean_outer_loss']))
            else:
                print('{}ways_{}shots_epoch{}:ACC {:.4f},LOSS {:.4f}'.format(num_ways, num_shots, e, results2['accuracies_after'], results2['mean_outer_loss']))
                with open(os.path.join(dirname, 'log_results.txt'), 'a+') as f:
                    f.write('{}ways_{}shots_epoch{}:ACC {:.4f},LOSS {:.4f}\n'.format(num_ways, num_shots, e, results2['accuracies_after'], results2['mean_outer_loss']))
 
def test(args):
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        with open(os.path.join(args.folder, 'config.json') , 'r') as f:
            config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder
    if args.test_num_steps > 0:
        config['num_steps'] = args.test_num_steps
    if args.test_num_batches > 0:
        config['num_batches'] = args.test_num_batches
    if args.test_num_shots > 0:
        config['num_shots'] = args.test_num_shots
    if args.model_path != '0':
        config['model_path']=args.model_path
    device = torch.device('cuda:{}'.format(args.cuda_no) if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')
    print(device)

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=config['num_shots'],
                                      num_test_per_class=config['num_shots_test'])
    if config['dataset'] == 'sinusoid':
        transform = ToTensor1D()
        meta_test_dataset = Sinusoid(config['num_shots'] + config['num_shots_test'],
            num_tasks=1000000, transform=transform, target_transform=transform,
            dataset_transform=dataset_transform)
        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif config['dataset'] == 'omniglot':
        transform = Compose([Resize(28), ToTensor()])
        meta_test_dataset = Omniglot(config['folder'], transform=transform,
            target_transform=Categorical(config['num_ways']),
            num_classes_per_task=config['num_ways'], meta_test=True,
            dataset_transform=dataset_transform, download=True)
        model = ModelConvOmniglot(config['num_ways'],
                                  hidden_size=config['hidden_size'])
        loss_function = F.cross_entropy

    elif config['dataset'] == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])
        meta_test_dataset = MiniImagenet(config['folder'], transform=transform,
            target_transform=Categorical(config['num_ways']),
            num_classes_per_task=config['num_ways'], meta_test=True,
            dataset_transform=dataset_transform, download=True)
        model = ModelConvMiniImagenet(config['num_ways'],
                                      hidden_size=config['hidden_size'])
        loss_function = F.cross_entropy
    elif config['dataset'] == 'tieredimagenet':
        transform = Compose([Resize(84), ToTensor()])
        meta_test_dataset = TieredImagenet(config['folder'], transform=transform,
            target_transform=Categorical(config['num_ways']),
            num_classes_per_task=config['num_ways'], meta_test=True,
            dataset_transform=dataset_transform, download=True)
        model = ModelConvMiniImagenet(config['num_ways'],
                                      hidden_size=config['hidden_size'])
        loss_function = F.cross_entropy
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(config['dataset']))

    with open(config['model_path'], 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    metalearner = ModelAgnosticMetaLearning(model,
        first_order=config['first_order'], num_adaptation_steps=config['num_steps'],
        step_size=config['step_size'], loss_function=loss_function, device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   desc='Test')
    print('results_{}_ways_{}_shots_model_{}: {}'.format(config['num_ways'], config['num_shots'],config['model_path'].split('/')[-1],results))
    # with open(os.path.join(args.output_folder, "log.txt"), 'a') as f:
    #     f.write('results_{}_ways_{}_shots_model_{}: {}\n'.format(config['num_ways'], config['num_shots'],config['model_path'].split('/')[-1],results))
    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'results_{}_ways_{}_shots_model_{}.json'.format(config['num_ways'], config['num_shots'], config['model_path'].split('/')[-1])), 'w') as f:
        json.dump(results, f)

import random
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=int, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')

    # Optimization
    parser.add_argument('--test-num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--test-num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--cuda-no', type=int, default=0, help='Number of cuda')
    parser.add_argument('--seed', type=int, default=123456, help='random seed')

    parser.add_argument('--test-num-shots', type=int, default=-1,
                        help='Number of fast adaptation steps, ie. gradient descent updates '
                             '(default: number of steps in configuration file).')

    parser.add_argument('--model-path', type=str, default='0', help='model path.')
    
    parser.add_argument('--best-only',type=bool, default=True, help='only test best model.')

    args = parser.parse_args()

    seed_torch(args.seed)
    main(args)
