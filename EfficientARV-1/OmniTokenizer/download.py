import requests
from tqdm import tqdm
import os
import torch

from .omnitokenizer import VQGAN as OmniTokenizer_VQGAN
from .lm_transformer import Net2NetTransformer

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(id, fname, root=os.path.expanduser('./ckpts')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


def load_vqgan(tokenizer, vqgan_ckpt, device=torch.device('cpu')):
    vqgan = OmniTokenizer_VQGAN.load_from_checkpoint(vqgan_ckpt, strict=False).to(device)
    print(f"Load VQGAN weights from {vqgan_ckpt}.")
    vqgan.eval()

    return vqgan

def load_transformer(gpt_ckpt, vqgan_ckpt, stft_vqgan_ckpt='', device=torch.device('cpu')):
    gpt = Net2NetTransformer.load_from_checkpoint(gpt_ckpt, strict=False, vqvae_ckpt=vqgan_ckpt).to(device)
    print(f"Load Transformer weights from {gpt_ckpt}.")
    gpt.eval()

    return gpt


_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'

def load_i3d_pretrained(device=torch.device('cpu')):
    from .fvd.pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d
