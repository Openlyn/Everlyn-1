#!/bin/bash
#SBATCH --job-name=FFHQ-2
#SBATCH -p q_intel_gpu_nvidia_h20_6 
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 16
#SBATCH -w gpu016
#SBATCH -o /online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/error/result_P2.out
#SBATCH -e /online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/FFHQ/error/error_P2.out

source activate /home/export/base/ycsc_xfangam/xfangam/.conda/envs/share_VAR

torchrun --standalone --nproc_per_node=4 --nnodes=1 --node_rank=0  train_wasserstein_quantizer.py --epochs=200 --ae_lr=5e-4 --std=0.05 --global_batch_size=128 --codebook_size=50000  --latent_reso=32 --latent_dim=4 --feature_dim=256 --alpha=0.2 --beta=0.2 --gamma=0.0 --lambd=1.0
torchrun --standalone --nproc_per_node=4 --nnodes=1 --node_rank=0  train_wasserstein_quantizer.py --epochs=200 --ae_lr=5e-4 --std=0.05 --global_batch_size=128 --codebook_size=100000 --latent_reso=32 --latent_dim=4 --feature_dim=256 --alpha=0.2 --beta=0.2 --gamma=0.0 --lambd=1.0