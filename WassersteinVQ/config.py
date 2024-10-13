import json
import os
import random
import re
import subprocess
import sys
import time
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Union
import argparse
import utils.dist as dist

def parse_arg():
    parser = argparse.ArgumentParser(description='Wasserstein Quantizer (Latent Dim vs Latent Resolution).') 

    ###Dataset and Dataloader Configuration
    parser.add_argument('--dataset_dir', default="/online1/ycsc_xfangam/xfangam/sunset/data/", type=str, help='the directory of dataset')
    parser.add_argument('--dataset_name', default='FFHQ', help='the name of dataset', choices=['ImageNet', 'FFHQ', 'JourneyDB'])
    parser.add_argument('--global_batch_size', type=int, default=32, help="the size of batch samples")
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--resize_rate', type=float, default=1.125, help="resize image to scale*resolution for image resolution>256.")
    parser.add_argument('--resolution', default=256, type=int, metavar='N', help='resolution of train and test')
    
    ###Model Configuration
    parser.add_argument('--ms_patch_size', default="1_2_3_4_5_6_8_10_13_16", type=str, help='multi-scale patch size.')
    parser.add_argument('--std', type=float, default=0.05, help="wasserstein distance between codebook vectors/feature vectors and  N(0, k^2 I), k is the std.")
    parser.add_argument('--max_patch_size', default=16, type=int, help='the maximum patch size.')
    parser.add_argument('--codebook_size', default=32768, type=int, help='the size of codebook.')
    parser.add_argument('--codebook_dim', default=16, type=int, help='the dimension of codebook vectors.')
    parser.add_argument('--vgg_checkpoint', default="/online1/ycsc_xfangam/xfangam/sunset/output/VGG", type=str, help='the directory of vgg checkpoint')
    parser.add_argument('--latent_reso', default=32, type=int, help='the resolution of latent variables.')
    parser.add_argument('--latent_dim', default=4, type=int, help='the dim of latent variables.')
    parser.add_argument('--feature_dim', default=256, type=int, help='the dim of encoder/decoder variables.')
    parser.add_argument('--ae_config_path', default="/online1/ycsc_xfangam/xfangam/sunset/model/yaml_files/vq-f16.yaml", type=str, help='the directory of yaml file.')

    ###Loss Configuration
    parser.add_argument('--alpha', type=float, default=0.2, help="the hyperparameter of vq_loss.")
    parser.add_argument('--beta', type=float, default=0.2, help=" the hyperparameter of commit_loss.")
    parser.add_argument('--gamma', type=float, default=0.2, help="the hyperparameter of wasserstein_loss.")
    parser.add_argument('--lambd', type=float, default=1.0, help="the hyperparameter of lpips_loss.")
    parser.add_argument("--rate_d", type=float, default=0.2, help="GAN Loss")

    ###Training Configuration
    parser.add_argument('--model_name', default='wasserstein_quantizer', help='the name of models.', choices=['wasserstein_quantizer'])
    parser.add_argument('--resume', action='store_true', help='reloading model from specified checkpoint.')
    parser.add_argument("--disc_start", default=50000, type=int, help="GAN Loss Start")
    parser.add_argument('--epochs', type=int, default=200, help="training epochs.")
    parser.add_argument('--seed', type=int, default=12, metavar='S', help='random seed (also job id).')
    parser.add_argument('--grad_clip', type=float, default=1.0, help="using gradient clip.")
    parser.add_argument('--ae_lr', default=5e-4, type=float, metavar='LR', help='initial (base) learning rate for encoder-decoder architecture.', dest='ae_lr')
    parser.add_argument('--weight_decay', default=0.05, type=float, metavar='W', help='weight decay.')
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N', help='number of warmup epochs.')
    parser.add_argument('--checkpoint_dir', default="/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/checkpoint/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--results_dir', default="/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/results/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--saver_dir', default="/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/saver/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--nnodes', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training.')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend.')

    ###Reconstruction and Generation Configuration
    parser.add_argument('--rec_image_dir', default="/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/rec_images/", type=str, help='the directory of reconstructed images.')
    parser.add_argument('--rec_results_dir', default="/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/rec_results/", type=str, help='the directory of reconstructed images.')
    parser.add_argument('--rec_name', default='Codebook-100000', help='the name of models.', choices=['Codebook-100000', 'Codebook-50000', 'Codebook-32768', 'Codebook-16384', 'Codebook-16384-no', 'Codebook-8192', 'Codebook-4096'])
    
    args = parser.parse_args()

    args.batch_size = round(args.global_batch_size  / dist.get_world_size())
    args.workers = min(max(0, args.workers), args.batch_size)
    args.world_size = dist.get_world_size()
    print("args.world_size:", args.world_size)

    if args.max_patch_size == 16:
        args.ms_patch_size = '1_2_3_4_5_6_8_10_13_16'
    elif args.max_patch_size == 32:
        args.ms_patch_size = '1_2_3_4_6_9_13_18_24_32'

    args.ms_token_size = tuple(map(int, args.ms_patch_size.replace('-', '_').split('_')))
    args.codebook_dim = int(args.latent_reso/16) * int(args.latent_reso/16) * args.latent_dim
    args.factor = int(args.latent_reso/16)

    args.data_pre = '{}'.format(args.dataset_name)
    args.model_pre = 'model_{}_{}_{}_{}_{}'.format(args.codebook_size, args.codebook_dim, args.latent_reso, args.latent_dim, args.feature_dim)
    args.loss_pre = 'loss_{}_{}_{}_{}_{}'.format(args.alpha, args.beta, args.gamma, args.lambd, args.rate_d)
    args.training_pre = '{}_{}_{}'.format(args.model_name, args.epochs, args.ae_lr)
    args.saver_name_pre = args.training_pre + '_' + args.data_pre + '_' + args.model_pre + '_' + args.loss_pre
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    return args