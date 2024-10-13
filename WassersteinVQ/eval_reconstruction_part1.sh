#!/bin/bash
#SBATCH --job-name=ImageNet-rec
#SBATCH -p q_intel_gpu_nvidia_h20_6 
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -c 12
#SBATCH -w gpu016
#SBATCH -o /online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/ImageNet-1k/rec_error/result_P1.out
#SBATCH -e /online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/ImageNet-1k/rec_error/error_P1.out

source activate /home/export/base/ycsc_xfangam/xfangam/.conda/envs/share_VAR

torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0  eval_reconstruction.py --rec_name=Codebook-100000 --epochs=20 --ae_lr=5e-4 --std=0.05 --global_batch_size=256 --codebook_size=100000 --latent_reso=32 --latent_dim=4 --feature_dim=256 --alpha=0.2 --beta=0.2 --gamma=0.3 --lambd=1.0