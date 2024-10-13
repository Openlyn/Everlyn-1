import gc
import os
import shutil
import sys
import time
import warnings
import numpy as np
import torch
from torch import nn, optim
import math
import json
import random
import scipy.io as sio
from torch.nn import functional as F
from scipy.io import savemat
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from data.dataloader import build_dataloader
import torchvision.models as torchvision_models
from torchvision import models, datasets, transforms
from utils import dist
from torch import distributed as tdist

import config
from utils.util import Logger, LossManager, Pack
from data import dataloader
from model.vqvae import VQVAE
from metric.metric import PSNR, LPIPS, SSIM

def main_worker(args):
    torch.cuda.set_device(dist.get_local_rank())
    model = VQVAE(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(dist.get_local_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=True)
    
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    checkpoint_name = "checkpoint-G1P1_pixelshuffle_quantizer_50_5e-05_FFHQ_model_256_0.05_128_4_S1_loss_0.05_0.2_0.3_0.3_0.0_1.0-resume.pth.tar"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    loc = 'cuda:{}'.format(dist.get_local_rank())
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    args = checkpoint['args']

    train_dataloader, val_dataloader, train_sampler, len_train_set, len_val_set = build_dataloader(args)
    model.eval()

    ssim, psnr, lpips, rec_loss_scalar, wasserstein_distance_scalar, codebook_utilization, perplexity, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    quant_error  = [0.0 for i in range(len(args.ms_token_size))]
    commit_error = [0.0 for i in range(len(args.ms_token_size))]
    wasser_error = [0.0 for i in range(len(args.ms_token_size))]

    results = {'psnr':[], 'ssim':[], 'lpips': [], 'codebook_utilization':[], 'perplexity':[], 'rec_loss':[], 'wasserstein_distance':[], \
        'quant_1':[], 'quant_2':[], 'quant_3':[], 'quant_4':[], 'quant_5':[], 'quant_6':[], 'quant_7':[], 'quant_8':[], 'quant_9':[], 'quant_10':[], \
        'commit_1':[], 'commit_2':[], 'commit_3':[], 'commit_4':[], 'commit_5':[], 'commit_6':[], 'commit_7':[], 'commit_8':[], 'commit_9':[], 'commit_10':[], \
        'wasser_1':[], 'wasser_2':[], 'wasser_3':[], 'wasser_4':[], 'wasser_5':[], 'wasser_6':[], 'wasser_7':[], 'wasser_8':[], 'wasser_9':[], 'wasser_10':[]}

    codebook_histogram_all: torch.Tensor = 0.0
    for step, (x, _) in enumerate(val_dataloader):
        print("step:[{}/{}]".format(step+1, len(val_dataloader)))
        x = x.cuda(dist.get_local_rank(), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size

        with torch.no_grad():
            x_rec, rec_loss, wasserstein_distance, codebook_histogram, level_quant_error, level_commit_error, level_wasser_error = model.module.collect_eval_info(x)
            codebook_histogram_all += codebook_histogram
            batch_lpips = lpips_metric(x, x_rec).sum()
            
            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0

            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        if dist.initialized(): 
            handler1 = tdist.all_reduce(batch_lpips, async_op=True)
            handler2 = tdist.all_reduce(batch_psnr, async_op=True)
            handler3 = tdist.all_reduce(batch_ssim, async_op=True)
            handler1.wait()
            handler2.wait()
            handler3.wait()

        if dist.is_local_master():
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            wasserstein_distance_scalar += wasserstein_distance.item() * batch_size
            rec_loss_scalar += rec_loss.item() * batch_size
            
            for i in range(len(args.ms_token_size)):
                quant_error[i]  += level_quant_error[i].data.cpu().item() * batch_size
                commit_error[i] += level_commit_error[i].data.cpu().item() * batch_size
                wasser_error[i] += level_wasser_error[i].data.cpu().item() * batch_size

    codebook_usage_counts = (codebook_histogram_all > 0).float().sum()
    codebook_utilization = codebook_usage_counts.item() / args.codebook_size

    avg_probs = codebook_histogram_all/codebook_histogram_all.sum(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    if dist.is_local_master():
        results['psnr'].append(psnr/len_val_set)
        results['ssim'].append(ssim/len_val_set)
        results['lpips'].append(lpips/len_val_set)
        results['codebook_utilization'].append(codebook_utilization)
        results['perplexity'].append(perplexity.item())
        results['rec_loss'].append(rec_loss_scalar/total_num)
        results['wasserstein_distance'].append(wasserstein_distance_scalar/total_num)

        for i in range(len(args.ms_token_size)):
            quant_error[i] = quant_error[i]/total_num
            commit_error[i] = commit_error[i]/total_num
            wasser_error[i] = wasser_error[i]/total_num

        results['quant_1'].append(quant_error[0])
        results['quant_2'].append(quant_error[1])
        results['quant_3'].append(quant_error[2])
        results['quant_4'].append(quant_error[3])
        results['quant_5'].append(quant_error[4])
        results['quant_6'].append(quant_error[5])
        results['quant_7'].append(quant_error[6])
        results['quant_8'].append(quant_error[7])
        results['quant_9'].append(quant_error[8])
        results['quant_10'].append(quant_error[9])

        results['commit_1'].append(commit_error[0])
        results['commit_2'].append(commit_error[1])
        results['commit_3'].append(commit_error[2])
        results['commit_4'].append(commit_error[3])
        results['commit_5'].append(commit_error[4])
        results['commit_6'].append(commit_error[5])
        results['commit_7'].append(commit_error[6])
        results['commit_8'].append(commit_error[7])
        results['commit_9'].append(commit_error[8])
        results['commit_10'].append(commit_error[9])

        results['wasser_1'].append(wasser_error[0])
        results['wasser_2'].append(wasser_error[1])
        results['wasser_3'].append(wasser_error[2])
        results['wasser_4'].append(wasser_error[3])
        results['wasser_5'].append(wasser_error[4])
        results['wasser_6'].append(wasser_error[5])
        results['wasser_7'].append(wasser_error[6])
        results['wasser_8'].append(wasser_error[7])
        results['wasser_9'].append(wasser_error[8])
        results['wasser_10'].append(wasser_error[9])

        #save reconstruction_performance results
        data_frame = pd.DataFrame(data=results, index=range(1, 1 + 1))
        data_frame.to_csv('{}/eval_{}_rec_results.csv'.format(args.results_dir, args.saver_name_pre), index_label='epoch')

if __name__ == '__main__':
    dist.initialize(fork=False, timeout=30)
    dist.barrier()
    args = config.parse_arg()
    dict_args = vars(args)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    if dist.is_local_master():
        for k, v in zip(dict_args.keys(), dict_args.values()):
            print("{0}: {1}".format(k, v))

    main_worker(args)  


