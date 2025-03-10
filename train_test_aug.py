import argparse, os
from train_un_maml_pem_aug import train_un_maml_pem_aug
from test import test, test_all, seed_torch

def main(args):
    train_un_maml_pem_aug(args)

    seed_torch(args.seed)
    # test(args)
    test_all(args)
    args.test_num_shots = 5
    # test(args)
    # test_all(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAML')

    # train
    
    # General
    parser.add_argument('folder', type=str, 
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet','tieredimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default='./tmp',
        help='Path to the output folder to save the model.')
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
    parser.add_argument('--num-workers', type=int, default=8,
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
    
    #test
    parser.add_argument('--config', type=str,
        help='Path to the configuration file returned by `train.py`.')

    parser.add_argument('--test-num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--test-num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')

    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed')

    parser.add_argument('--test-num-shots', type=int, default=-1,
                        help='Number of fast adaptation steps, ie. gradient descent updates '
                             '(default: number of steps in configuration file).')

    parser.add_argument('--model-path', type=str, default='0',
                        help='model path.')
    
    parser.add_argument('--best-only',type=bool, default=True, help='only test best model.')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
