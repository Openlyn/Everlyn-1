## reference from "https://github.com/zh460045050/VQGAN-LC/tree/main?tab=readme-ov-file#ldm-training"
#from cleanfid import fid
import pyiqa
import torch
import torch.nn as nn
import piq

## metrics for reconstruction evaluation 
## rFID LPIPS PSNR SSIM, Codebook utilization

## rFID is calculated by cleanfid (https://github.com/GaParmar/clean-fid). pip install clean-fid
#def calc_frechet_distance(fdir1, fdir2, mode="clean"):
#    return fid.compute_fid(fdir1, fdir2, mode="clean")

## PSNR and LPIPS are computed by pyiqa (https://github.com/chaofengc/IQA-PyTorch). pip install pyiqa
###### data range (0, 1) 
class PSNR():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)

###### data range (-1, 1) 
class LPIPS():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('lpips', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)

###### data range (0, 1)   
class SSIM():
    #def __init__(self, device=None):
    #self.iqa_metric = pyiqa.create_metric('ssim', device=device)
    
    def __call__(self, real, fake):
        return piq.ssim(real, fake, data_range=1., reduction='none') #self.iqa_metric(real, fake)


## metrics for generation evaluation 
## Fréchet inception distance (FID), inception score (IS)
if __name__ == "__main__":
    import torch
    lpips = LPIPS('cuda:0')
    psnr = PSNR('cuda:0')
    ssim = SSIM('cuda:0')
    usages = CodebookUtilization(10, 4096)

    real = torch.randn(32, 3, 256, 256).cuda()
    fake = torch.randn(32, 3, 256, 256)

    print(f'{lpips(real, fake)=}')
    print(f'{psnr(real, fake)=}')
    print(f'{ssim(real, fake)=}')

    for ii in range(10):
        codes = torch.randint(0, 4096, size=(10, 256))
        print(f'{usages(codes)=}')