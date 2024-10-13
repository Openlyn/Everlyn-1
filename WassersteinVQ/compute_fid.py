from cleanfid import fid
import pickle

rec_image_path = "/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/ImageNet-1k/rec_images/Codebook-100000/epoch-20/Rec"
org_image_path = "/online1/ycsc_xfangam/xfangam/sunset/output/wasserstein_quantizer/ImageNet-1k/rec_images/Codebook-100000/epoch-20/Org"

fid = fid.compute_fid(org_image_path, rec_image_path)

print('rFID1:{}'.format(fid))