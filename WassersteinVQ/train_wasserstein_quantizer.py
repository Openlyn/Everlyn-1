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
from utils.util import NativeScalerWithGradNormCount as NativeScaler

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate
from data import dataloader
from model.vqvae import VQVAE
from metric.metric import PSNR, LPIPS, SSIM

## calculation (codebook_utilization, wasserstein distance, level_quantization_error)
## (rec_loss, PSNR, SSIM)
def eval_one_epoch(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    
    ssim, psnr, lpips, rec_loss_scalar, wasserstein_distance_scalar, codebook_utilization, perplexity, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    quant_error  = [0.0 for i in range(len(args.ms_token_size))]
    commit_error = [0.0 for i in range(len(args.ms_token_size))]

    codebook_histogram_all: torch.Tensor = 0.0
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(dist.get_local_rank(), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size

        with torch.no_grad():
            x_rec, rec_loss, wasserstein_distance, codebook_histogram, level_quant_error, level_commit_error = model.module.collect_eval_info(x)
            
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
    
    codebook_usage_counts = (codebook_histogram_all > 0).float().sum()
    codebook_utilization  = codebook_usage_counts.item() / args.codebook_size

    avg_probs = codebook_histogram_all/codebook_histogram_all.sum(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_codebook_utilization = codebook_utilization
    eval_perplexity = perplexity.item()
    eval_rec_loss = rec_loss_scalar/total_num
    eval_wasserstein_distance = wasserstein_distance_scalar/total_num

    for i in range(len(args.ms_token_size)):
        quant_error[i]  = quant_error[i]/total_num
        commit_error[i] = commit_error[i]/total_num

    model.train()
    return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, codebook_utilization=eval_codebook_utilization, perplexity=eval_perplexity, rec_loss=eval_rec_loss, wasserstein_distance=eval_wasserstein_distance, quant_error=quant_error, commit_error=commit_error)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main_worker(args):
    torch.cuda.set_device(dist.get_local_rank())
    model = VQVAE(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(dist.get_local_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=True)

    ae_para = list(model.module.encoder.parameters()) + list(model.module.decoder.parameters()) + list(model.module.quant_conv.parameters()) + list(model.module.post_quant_conv.parameters())
    opt_ae = torch.optim.Adam(ae_para, lr=args.ae_lr, betas=(0.5, 0.9), eps=1e-7)

    disc_para = list(model.module.discriminator.parameters())
    opt_disc = torch.optim.Adam(disc_para, lr=args.ae_lr, betas=(0.5, 0.9), eps=1e-7)

    loss_scaler_ae = NativeScaler()
    loss_scaler_disc = NativeScaler()

    train_dataloader, val_dataloader, train_sampler, len_train_set, len_val_set = build_dataloader(args)
    start_epoch = 1
    if args.resume:
        print("reloading model...")
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+ args.saver_name_pre +'-resume'+'.pth.tar')
        loc = 'cuda:{}'.format(dist.get_local_rank())
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        opt_ae.load_state_dict(checkpoint['opt_ae'])
        opt_disc.load_state_dict(checkpoint['opt_disc'])
        loss_scaler_ae.load_state_dict(checkpoint["scaler_ae"])
        loss_scaler_disc.load_state_dict(checkpoint["scaler_disc"])
        args = checkpoint['args']

        if dist.is_local_master():
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

    results = {'ae_loss':[], 'rec_loss': [], 'commit_loss':[], 'vq_loss':[], 'lpips_loss':[], 'wasserstein_loss':[], 'd_loss':[], 'g_loss':[], 'perplexity':[], 'codebook_utilization':[], \
        'quant_1':[], 'quant_2':[], 'quant_3':[], 'quant_4':[], 'quant_5':[], 'quant_6':[], 'quant_7':[], 'quant_8':[], 'quant_9':[], 'quant_10':[]}
    
    results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'lpips': [], 'codebook_utilization':[], 'perplexity':[], 'rec_loss':[], 'wasserstein_distance':[], \
        'quant_1':[], 'quant_2':[], 'quant_3':[], 'quant_4':[], 'quant_5':[], 'quant_6':[], 'quant_7':[], 'quant_8':[], 'quant_9':[], 'quant_10':[], \
        'commit_1':[], 'commit_2':[], 'commit_3':[], 'commit_4':[], 'commit_5':[], 'commit_6':[], 'commit_7':[], 'commit_8':[], 'commit_9':[], 'commit_10':[]}
    
    train_loss = LossManager()
    best_psnr, current_psnr = 0.0, 0.0
    results_val_index = 1
    print("Start training...")
    for epoch in range(start_epoch, args.epochs+1):
        train_sampler.set_epoch(epoch)
        print("epoch:%d, ae_lr:%4f"%(epoch, opt_ae.param_groups[0]["lr"]))
        print("epoch:%d, disc_lr:%4f"%(epoch, opt_disc.param_groups[0]["lr"]))

        iters_per_epoch = len(train_dataloader) 
        ae_loss_scalar, rec_loss, vq_loss, commit_loss, lpips_loss, wasserstein_loss, d_loss_scalar, g_loss, perplexity, codebook_utilization, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        level_quant_error = [0.0 for i in range(len(args.ms_token_size))]

        model.train()
        start_time = time.time()
        for step, (x, _) in enumerate(train_dataloader):
            cur_iter = len(train_dataloader) * (epoch-1) + step
            cur_epoch = cur_iter/len(train_dataloader)

            x = x.cuda(dist.get_local_rank(), non_blocking=True)
            batch_size = x.size(0)

            ae_loss, loss_pack, level_quantization_error = model(x, cur_iter, step=0)
            opt_ae.zero_grad()
            adjust_learning_rate(opt_ae, cur_epoch, args)
            loss_scaler_ae(ae_loss, opt_ae, parameters=ae_para, update_grad=True)

            if cur_iter > args.disc_start:
                d_loss, loss_pack2 = model(x, cur_iter, step=1)
                opt_disc.zero_grad()
                adjust_learning_rate(opt_disc, cur_epoch, args)
                loss_scaler_disc(d_loss, opt_disc, parameters=disc_para, update_grad=True)
                loss_pack.add(loss_pack2)
            else:
                d_loss = torch.zeros(1)

            torch.cuda.synchronize()            
            train_loss.add_loss(loss_pack)
            if dist.is_local_master():
                total_num += batch_size
                ae_loss_scalar += loss_pack.ae_loss.item() * batch_size
                d_loss_scalar += d_loss.item() * batch_size
                rec_loss += loss_pack.rec_loss.item() * batch_size
                commit_loss += loss_pack.commit_loss.item() * batch_size
                vq_loss += loss_pack.vq_loss.item() * batch_size
                g_loss += loss_pack.g_loss.item() * batch_size
                lpips_loss += loss_pack.lpips_loss.item() * batch_size
                wasserstein_loss += loss_pack.wasserstein_loss.item() * batch_size
                perplexity += loss_pack.perplexity.item() * batch_size
                codebook_utilization += loss_pack.codebook_utilization * batch_size

                for i in range(len(args.ms_token_size)):
                    level_quant_error[i] += level_quantization_error[i].cpu().item() * batch_size

            if dist.is_local_master() and (step+1) %10 ==0:
                print(train_loss.pprint(window=50, prefix='Train Epoch: [{}/{}] Iters:[{}/{}]'.format(epoch, args.epochs, step+1, len(train_dataloader))))

        train_loss.clear()
        ######################### start conducting statistical analysis per epoch on training dataset ##########
        print("######### start conducting statistical analysis per epoch on training dataset #########")
        if dist.is_local_master():
            results['ae_loss'].append(ae_loss_scalar/total_num)
            results['rec_loss'].append(rec_loss/total_num)
            results['commit_loss'].append(commit_loss/total_num)
            results['vq_loss'].append(vq_loss/total_num)
            results['lpips_loss'].append(lpips_loss/total_num)
            results['d_loss'].append(d_loss_scalar/total_num)
            results['g_loss'].append(g_loss/total_num)
            results['wasserstein_loss'].append(wasserstein_loss/total_num)
            results['perplexity'].append(perplexity/total_num)
            results['codebook_utilization'].append(codebook_utilization/total_num)

            for i in range(len(args.ms_token_size)):
                level_quant_error[i] = level_quant_error[i]/total_num

            results['quant_1'].append(level_quant_error[0])
            results['quant_2'].append(level_quant_error[1])
            results['quant_3'].append(level_quant_error[2])
            results['quant_4'].append(level_quant_error[3])
            results['quant_5'].append(level_quant_error[4])
            results['quant_6'].append(level_quant_error[5])
            results['quant_7'].append(level_quant_error[6])
            results['quant_8'].append(level_quant_error[7])
            results['quant_9'].append(level_quant_error[8])
            results['quant_10'].append(level_quant_error[9])
            #save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('{}/train_{}_statistics.csv'.format(args.results_dir, args.saver_name_pre), index_label='epoch')

        print("######### save checkpoint of each epoch #########")
        if dist.is_local_master() and epoch%10 == 0:
            model.train()
            checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'-'+str(epoch)+'.pth.tar')
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'opt_ae': opt_ae.state_dict(), 'opt_disc': opt_disc.state_dict(), 'scaler_ae': loss_scaler_ae.state_dict(), 'scaler_disc': loss_scaler_disc.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path)

        if epoch%10 == 0:
            print("######### start evaluation per 10 epoch #########")
            with torch.no_grad():
                results_pack = eval_one_epoch(args, model, epoch, val_dataloader, len_val_set)

        if dist.is_local_master() and epoch%10 == 0:
            results_eval['epoch'].append(epoch)
            results_eval['psnr'].append(results_pack.psnr)
            results_eval['ssim'].append(results_pack.ssim)
            results_eval['lpips'].append(results_pack.lpips)
            results_eval['codebook_utilization'].append(results_pack.codebook_utilization)
            results_eval['perplexity'].append(results_pack.perplexity)
            results_eval['rec_loss'].append(results_pack.rec_loss)
            results_eval['wasserstein_distance'].append(results_pack.wasserstein_distance)

            results_eval['quant_1'].append(results_pack.quant_error[0])
            results_eval['quant_2'].append(results_pack.quant_error[1])
            results_eval['quant_3'].append(results_pack.quant_error[2])
            results_eval['quant_4'].append(results_pack.quant_error[3])
            results_eval['quant_5'].append(results_pack.quant_error[4])
            results_eval['quant_6'].append(results_pack.quant_error[5])
            results_eval['quant_7'].append(results_pack.quant_error[6])
            results_eval['quant_8'].append(results_pack.quant_error[7])
            results_eval['quant_9'].append(results_pack.quant_error[8])
            results_eval['quant_10'].append(results_pack.quant_error[9])

            results_eval['commit_1'].append(results_pack.commit_error[0])
            results_eval['commit_2'].append(results_pack.commit_error[1])
            results_eval['commit_3'].append(results_pack.commit_error[2])
            results_eval['commit_4'].append(results_pack.commit_error[3])
            results_eval['commit_5'].append(results_pack.commit_error[4])
            results_eval['commit_6'].append(results_pack.commit_error[5])
            results_eval['commit_7'].append(results_pack.commit_error[6])
            results_eval['commit_8'].append(results_pack.commit_error[7])
            results_eval['commit_9'].append(results_pack.commit_error[8])
            results_eval['commit_10'].append(results_pack.commit_error[9])

            #save reconstruction_performance results
            data_frame = pd.DataFrame(data=results_eval, index=range(1, results_val_index+1))
            data_frame.to_csv('{}/eval_{}_rec_results.csv'.format(args.results_dir, args.saver_name_pre), index_label='index')
            results_val_index += 1
        ############################## start evaluation per epoch

            current_psnr = results_pack.psnr
            if current_psnr >= best_psnr:
                model.train()
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'-best'+'.pth.tar')    
                save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'opt_ae': opt_ae.state_dict(), 'opt_disc': opt_disc.state_dict(), 'scaler_ae': loss_scaler_ae.state_dict(), 'scaler_disc': loss_scaler_disc.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path)
        
                best_psnr = max(best_psnr, current_psnr)
                print("best_psnr:{}, current_psnr:{}".format(best_psnr, current_psnr))
    
    print("######### start saving final checkpoint #########")
    model.train()
    if dist.is_local_master():
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'-final'+'.pth.tar')
        save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'opt_ae': opt_ae.state_dict(), 'opt_disc': opt_disc.state_dict(), 'scaler_ae': loss_scaler_ae.state_dict(), 'scaler_disc': loss_scaler_disc.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path)

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