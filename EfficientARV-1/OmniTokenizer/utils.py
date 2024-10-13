import torch
import imageio

import math
import numpy as np

import sys
import pdb as pdb_original


def inflate_gen(state_dict, temporal_patch_size, spatial_patch_size, strategy="average", inflation_pe=False):
    new_state_dict = state_dict.copy()

    pe_image0_w = state_dict["encoder.to_patch_emb_first_frame.1.weight"] # image_channel * patch_width * patch_height
    pe_image0_b = state_dict["encoder.to_patch_emb_first_frame.1.bias"] # image_channel * patch_width * patch_height
    pe_image1_w = state_dict["encoder.to_patch_emb_first_frame.2.weight"] # image_channel * patch_width * patch_height, dim
    pe_image1_b = state_dict["encoder.to_patch_emb_first_frame.2.bias"] # image_channel * patch_width * patch_height
    pe_image2_w = state_dict["encoder.to_patch_emb_first_frame.3.weight"] # image_channel * patch_width * patch_height
    pe_image2_b = state_dict["encoder.to_patch_emb_first_frame.3.bias"] # image_channel * patch_width * patch_height

    pd_image0_w = state_dict["decoder.to_pixels_first_frame.0.weight"] # dim, image_channel * patch_width * patch_height
    pd_image0_b = state_dict["decoder.to_pixels_first_frame.0.bias"] # image_channel * patch_width * patch_height
    pe_video0_w = state_dict["encoder.to_patch_emb.1.weight"]

    if strategy == "average":
        pe_video0_w = torch.cat([pe_image0_w/temporal_patch_size] * temporal_patch_size)
        pe_video0_b = torch.cat([pe_image0_b/temporal_patch_size] * temporal_patch_size)

        pe_video1_w = torch.cat([pe_image1_w/temporal_patch_size] * temporal_patch_size, dim=-1)
        pe_video1_b = pe_image1_b # torch.cat([pe_image1_b/temporal_patch_size] * temporal_patch_size)

        pe_video2_w = pe_image2_w # torch.cat([pe_image2_w/temporal_patch_size] * temporal_patch_size)
        pe_video2_b = pe_image2_b # torch.cat([pe_image2_b/temporal_patch_size] * temporal_patch_size)

    elif strategy == "first":
        pe_video0_w = torch.cat([pe_image0_w] + [torch.zeros_like(pe_image0_w, dtype=pe_image0_w.dtype)] * (temporal_patch_size - 1))
        pe_video0_b = torch.cat([pe_image0_b] + [torch.zeros_like(pe_image0_b, dtype=pe_image0_b.dtype)] * (temporal_patch_size - 1))

        pe_video1_w = torch.cat([pe_image1_w] + [torch.zeros_like(pe_image1_w, dtype=pe_image1_w.dtype)] * (temporal_patch_size - 1), dim=-1)
        pe_video1_b = pe_image1_b # torch.cat([pe_image1_b] + [torch.zeros_like(pe_image1_b, dtype=pe_image1_b.dtype)] * (temporal_patch_size - 1))

        pe_video2_w = pe_image2_w # torch.cat([pe_image2_w] + [torch.zeros_like(pe_image2_w, dtype=pe_image2_w.dtype)] * (temporal_patch_size - 1))
        pe_video2_b = pe_image2_b # torch.cat([pe_image2_b] + [torch.zeros_like(pe_image2_b, dtype=pe_image2_b.dtype)] * (temporal_patch_size - 1))
    

    else:
        raise NotImplementedError

    
    new_state_dict["encoder.to_patch_emb.1.weight"] = pe_video0_w
    new_state_dict["encoder.to_patch_emb.1.bias"] = pe_video0_b

    new_state_dict["encoder.to_patch_emb.2.weight"] = pe_video1_w
    new_state_dict["encoder.to_patch_emb.2.bias"] = pe_video1_b

    new_state_dict["encoder.to_patch_emb.3.weight"] = pe_video2_w
    new_state_dict["encoder.to_patch_emb.3.bias"] = pe_video2_b
    

    if strategy == "average":
        pd_video0_w = torch.cat([pd_image0_w/temporal_patch_size] * temporal_patch_size)
        pd_video0_b = torch.cat([pd_image0_b/temporal_patch_size] * temporal_patch_size)
    
    elif strategy == "first":
        pd_video0_w = torch.cat([pd_image0_w] + [torch.zeros_like(pd_image0_w, dtype=pd_image0_w.dtype)] * (temporal_patch_size - 1))
        pd_video0_b = torch.cat([pd_image0_b] + [torch.zeros_like(pd_image0_b, dtype=pd_image0_b.dtype)] * (temporal_patch_size - 1))

    else:
        raise NotImplementedError

    
    new_state_dict["decoder.to_pixels.0.weight"] = pd_video0_w
    new_state_dict["decoder.to_pixels.0.bias"] = pd_video0_b

    return new_state_dict


def inflate_dis(state_dict, strategy="center"):
    print("#" * 50)
    print(f"Initialize the video discriminator with {strategy}.")
    print("#" * 50)
    idis_weights = {k: v for k, v in state_dict.items() if "image_discriminator" in k}
    vids_weights = {k: v for k, v in state_dict.items() if "video_discriminator" in k}

    new_state_dict = state_dict.copy()
    for k in vids_weights.keys():
        del new_state_dict[k]
    

    for k in idis_weights.keys():
        new_k = "video_discriminator" + k[len("image_discriminator"):]
        if "weight" in k and new_state_dict[k].ndim == 4:
            old_weight = state_dict[k]
            if strategy == "average":
                new_weight = old_weight.unsqueeze(2).repeat(1, 1, 4, 1, 1) / 4
            elif strategy == "center":
                new_weight_ = old_weight# .unsqueeze(2) # O I 1 K K
                new_weight = torch.zeros((new_weight_.size(0), new_weight_.size(1), 4, new_weight_.size(2), new_weight_.size(3)), dtype=new_weight_.dtype)
                new_weight[:, :, 1] = new_weight_
                
            elif strategy == "first":
                new_weight_ = old_weight# .unsqueeze(2)
                new_weight = torch.zeros((new_weight_.size(0), new_weight_.size(1), 4, new_weight_.size(2), new_weight_.size(3)), dtype=new_weight_.dtype)
                new_weight[:, :, 0] = new_weight_

            elif strategy == "last":
                new_weight_ = old_weight# .unsqueeze(2)
                new_weight = torch.zeros((new_weight_.size(0), new_weight_.size(1), 4, new_weight_.size(2), new_weight_.size(3)), dtype=new_weight_.dtype)
                new_weight[:, :, -1] = new_weight_
            else:
                raise NotImplementedError
            
            new_state_dict[new_k] = new_weight
        
        elif "bias" in k:
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[new_k] = state_dict[k]


    return new_state_dict

    
class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight


def save_video_grid(video, fname, nrow=None, fps=6):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1).contiguous()

    video = (video.detach().cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 0
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    # print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    # skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    # print('saved videos to', fname)


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""
