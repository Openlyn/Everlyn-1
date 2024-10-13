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
import torchvision.transforms as transforms
from PIL import Image
import pickle

import config
from utils.util import Logger, LossManager, Pack
from data import dataloader
from model.vqvae import VQVAE
from metric.metric import PSNR, LPIPS, SSIM
from cleanfid import fid
import matplotlib.pyplot as plt

def main_worker(args, epoch):
    ### create directionary
    rec_path  = os.path.join(args.rec_image_dir, args.rec_name)
    epoch_path = os.path.join(rec_path, "epoch-"+str(epoch))
    rec_epoch_path = os.path.join(epoch_path, "Rec")
    org_epoch_path = os.path.join(epoch_path, "Org")

    os.makedirs(epoch_path, exist_ok=True)
    os.makedirs(rec_epoch_path, exist_ok=True)
    os.makedirs(org_epoch_path, exist_ok=True)

    rFID = 0

    torch.cuda.set_device(dist.get_local_rank())
    model = VQVAE(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(dist.get_local_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=True)
    
    checkpoint_name = 'checkpoint-'+args.saver_name_pre+'-'+str(epoch)+'.pth.tar'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    loc = 'cuda:{}'.format(dist.get_local_rank())
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])

    train_dataloader, val_dataloader, train_sampler, len_train_set, len_val_set = build_dataloader(args)
    model.eval()

    account = 0
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(dist.get_local_rank(), non_blocking=True)
        batch_size = x.size(0)

        with torch.no_grad():
            x_rec, _, _, _, _, _ = model.module.collect_eval_info(x)

            x_org = (x + 1.0)/2
            x_rrec = (x_rec + 1.0)/2

            for i in range(batch_size):
                plt.imsave(os.path.join(org_epoch_path, "org_"+str(dist.get_local_rank())+"_"+str(account+i)+".png"), np.uint8(x_org[i].detach().cpu().numpy().transpose(1, 2, 0) * 255))
                plt.imsave(os.path.join(rec_epoch_path, "rec_"+str(dist.get_local_rank())+"_"+str(account+i)+".png"), np.uint8(x_rrec[i].detach().cpu().numpy().transpose(1, 2, 0) * 255))
        
        account += batch_size

    fid1 = 0
    fid2 = 0
    if dist.is_local_master():
        fid1 = fid.compute_fid(org_epoch_path, rec_epoch_path)
        fid2 = fid.compute_fid(org_epoch_path, rec_epoch_path, mode="clean")
        print('rFID1:{}, rFID2:{}'.format(fid1, fid2))
    rFID = fid1
    return rFID

if __name__ == '__main__':
    dist.initialize(fork=False, timeout=15)
    dist.barrier()
    args = config.parse_arg()
    dict_args = vars(args)
    #sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    if dist.is_local_master():
        for k, v in zip(dict_args.keys(), dict_args.values()):
            print("{0}: {1}".format(k, v))

    os.makedirs(os.path.join(args.rec_image_dir, args.rec_name), exist_ok=True)
    os.makedirs(os.path.join(args.rec_results_dir, args.rec_name), exist_ok=True)

    print("local_rank:", dist.get_local_rank())
    
    epochs = [7, 20]
    results = {'epoch':[], 'rFID':[]}
    index = 1
    for epoch in epochs:
        rFID = main_worker(args, epoch) 
        print("epoch:{},  rFID:{}".format(epoch, rFID))

        if dist.is_local_master():
            results['epoch'] = epoch
            results['rFID'] = rFID

            data_frame = pd.DataFrame(data=results, index=range(1, index))
            data_frame.to_csv('{}/reconstruct_{}_results.csv'.format(args.rec_results_dir, args.rec_name), index_label='index')
        index += 1


    

