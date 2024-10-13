import argparse
import os
import random
import sys
import os

# Add the directory to sys.path
sys.path.append('/mnt/sda/feilongtang/Hallucination/SID')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json
from vcd_add_noise import add_diffusion_noise
import warnings
warnings.filterwarnings("ignore")
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
time = datetime.now().strftime('%m-%d-%H:%M')
print(time)

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="llava-1.5", help="model")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
# vision contrastive decoding
parser.add_argument("--noise-step", type=int, default=500)
parser.add_argument("--use-cd", action='store_true', default=False)
parser.add_argument("--use-icd", action='store_true', default=False)
parser.add_argument("--use-vcd", action='store_true', default=False)
parser.add_argument("--cd-alpha", type=float, default=1)
parser.add_argument("--cd-beta", type=float, default=0.1)
parser.add_argument("--sample-greedy", default=True)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data-path", type=str, default="", help="data path")
parser.add_argument("--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--num-workers", type=int, default=1, help="num workers")
# fast token merging
parser.add_argument("--use-fast-v", action='store_true', default=False)
parser.add_argument('--fast-v-inplace', type=bool, default=False)
parser.add_argument("--fast-v-attention-rank", type=int, default=5)
parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
parser.add_argument("--fast-v-agg-layer", type=int, default=2)
parser.add_argument('--fast-v-sys-length', default=None, type=int, help='the length of system prompt')
parser.add_argument('--fast-v-image-token-length', default=None, type=int, help='the length of image token')
# opera-beamsearch
parser.add_argument("--test-sample", type=int, default=500)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", type=bool, default=False)
parser.add_argument("--scale-factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num-attn-candidates", type=int, default=5)
parser.add_argument("--penalty-weights", type=float, default=1.0)
parser.add_argument("--opera", action='store_true', default=False)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)

# set model decoding config
if args.model == "instructblip":
    if args.use_fast_v == True:
        model.llm_model.config.use_fast_v = args.use_fast_v
        model.llm_model.config.fast_v_inplace = args.fast_v_inplace
        model.llm_model.config.fast_v_sys_length = args.fast_v_sys_length
        model.llm_model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.llm_model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.llm_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
        model.llm_model.config.fast_v_agg_layer = args.fast_v_agg_layer
    else:
        model.llm_model.config.use_fast_v = args.use_fast_v
    model.llm_model.model.reset_fastv()
else:
    if args.use_fast_v == True:
        model.llama_model.config.use_fast_v = args.use_fast_v
        model.llama_model.config.fast_v_inplace = args.fast_v_inplace
        model.llama_model.config.fast_v_sys_length = args.fast_v_sys_length
        model.llama_model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.llama_model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.llama_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
        model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
    else:
        model.llama_model.config.use_fast_v = args.use_fast_v
    model.llama_model.model.reset_fastv()


processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


img_files = os.listdir(args.data_path)
random.shuffle(img_files)
img_files[1000]
print(img_files[1000])
with open('/mnt/sda/feilongtang/Hallucination/datasets/mscoco/annotations/instances_val2014.json', 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

img_dict = {}

categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )


# base_dir  = "./log/" + args.model
# if not os.path.exists(base_dir):
#     os.mkdir(base_dir)


base_dir = '/mnt/sda/feilongtang/Hallucination/SID/results/test_w/1'

# for img_id in tqdm(range(len(img_files))):
for img_id in range(len(img_files)):
    # print("img_id",img_id)
    if img_id == args.test_sample:
        break
    img_file = img_files[img_id]
    
    img_id = int(img_file.split(".jpg")[0][-6:])
    # print(img_id)
    if img_id == 15883:
        # print(img_id)
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id

        image_path = args.data_path + "/" + img_file
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image)
        image = image.unsqueeze(0)
        image = image.to(device)

        qu = "Please describe this image in detail."
        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", qu)

        if args.use_icd:
            text_cd = 'You are a confused image caption model.'
            if args.model == 'shikra':
                prompt_cd = qu[0].split("<im_end>")[0] + "<im_end>" + ' ' + text_cd + qu[0].split("<im_end>")[-1]
            elif args.model == 'llava-1.5' or args.model == 'instructblip':
                prompt_cd = qu[0].split("<ImageHere>")[0] + "<ImageHere>" + ' ' + text_cd + qu[0].split("<ImageHere>")[-1]
            # elif args.model == 'lrv_instruct' or args.model == 'minigpt4':
            else:
                prompt_cd = qu[0].split("</Img>")[0] + "</Img>" + ' ' + text_cd + qu[0].split("</Img>")[-1]

        else:
            text_cd = None

        
        if args.use_cd:
            image_cd = norm(image).to(device)
        elif args.use_vcd:
            image_cd = add_diffusion_noise(norm(image), args.noise_step)
            image_cd = norm(image_cd).to(device)
        else:
            image_cd = None
        
        
        with torch.inference_mode():
            with torch.no_grad():
                out = model.generate(
                    # {"image": norm(image), "prompt":qu},
                    prompt = qu,
                    image = norm(image).half(),
                    images_cd=(image_cd.half() if image_cd is not None else None),
                    prompt_cd =(prompt_cd if text_cd is not None else None),
                    use_nucleus_sampling=args.sample, 
                    num_beams=args.beam,
                    max_new_tokens=512,
                    output_attentions=True,
                    opera_decoding=args.opera,
                    scale_factor=args.scale_factor,
                    threshold=args.threshold,
                    num_attn_candidates=args.num_attn_candidates,
                    penalty_weights=args.penalty_weights,
                    use_cache=True,
                    sample_greedy = args.sample_greedy,
                    # do_sample=True,
                )
        img_save["caption"] = out[0]

        with open(os.path.join(base_dir, 'llava-{}samples-greedy.jsonl'.format(args.test_sample)), "a") as f:
                    json.dump(img_save, f)
                    f.write('\n')

    # # dump metric file
    # if args.use_fast_v == True and args.use_cd == True:
    #     with open(os.path.join(base_dir, 'top_important_ours-{}samples-cd-layer{}-token{}-time{}-greedy.jsonl'.format(args.test_sample, args.fast_v_agg_layer, args.fast_v_attention_rank, time)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')
    # elif args.use_vcd == True:
    #     with open(os.path.join(base_dir, 'degraded_ours-{}samples-vcd-sampling.jsonl'.format(args.test_sample)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')
    # elif args.use_icd == True:
    #     with open(os.path.join(base_dir, 'degraded_ours-{}samples-icd-sampling.jsonl'.format(args.test_sample)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')
    # else:
    #     with open(os.path.join(base_dir, 'ours-{}samples-opera.jsonl'.format(args.test_sample)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')

    


