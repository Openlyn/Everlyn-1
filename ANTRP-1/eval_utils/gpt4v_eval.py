import base64
import requests
from PIL import Image
from io import BytesIO

import re

import argparse
import os
import random
import sys
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

# from PIL import Image
from torchvision.utils import save_image
from vcd_add_noise import add_diffusion_noise
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json

# python pope_eval.py --model llava-1.5 --data_path /home/hfs/e/llm/mscoco/ --pope-type random --gpu-id 0 --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1



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




GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinationsshould be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Don't

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''


# OpenAI API Key
API_KEY = "sk-zk25a4367a3b700566928ae26397ed208e066f5adc192bbc"



def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True




def call_api(prompt, image_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.zhizengzeng.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json().keys())
    return response.json()


def get_gpt4v_answer(prompt, image_path):
    while 1:
        try:
            res = call_api(prompt, image_path)
            if "choices" in res.keys():
                return res["choices"][0]["message"]["content"]
            else:
                assert False
        except Exception as e:
            print("retry")
            # pass
    # return call_api(prompt, image_path)


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="shikra", help="model")
parser.add_argument("--pope-type", type=str, default="coco_popular", help="model")
parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
# vision contrastive decoding
parser.add_argument("--noise_step", type=int, default=500)
parser.add_argument("--use-cd", action='store_true', default=False)
parser.add_argument("--use-icd", action='store_true', default=False)
parser.add_argument("--use-vcd", action='store_true', default=False)
parser.add_argument("--cd-alpha", type=float, default=1)
parser.add_argument("--cd-beta", type=float, default=0.1)
# fast token merging
parser.add_argument("--use-fast-v", action='store_true', default=False)
parser.add_argument("--fast-v-inplace", default=False)
parser.add_argument("--fast-v-attention-rank", type=int, default=50)
parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
parser.add_argument("--fast-v-agg-layer", type=int, default=2)
# parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
# parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
# parser.add_argument("--data-path", type=str, default="/apdcephfs/private_huofushuo/dataset/mscoco/GQA/images", help="data path")
parser.add_argument("--data-path", type=str, default="/mnt/sda/feilongtang/Hallucination/datasets/mscoco/val2014/", help="data path")
parser.add_argument("--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=1, help="num workers")
parser.add_argument("--answers-file", type=str, default="/home/hfs/llm/OPERA-main/log/llava-1.5/pope/")
# auto-generation
parser.add_argument("--fast-v-sys-length", default=None, type=int, help='the length of system prompt')
parser.add_argument("--fast-v-image-token-length", default=None, type=int, help='the length of image token')
# opera-beamsearch 
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", default=True)
parser.add_argument("--scale-factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn-candidates", type=int, default=5)
parser.add_argument("--penalty-weights", type=float, default=1.0)
parser.add_argument("--opera", action='store_true',  default=False)
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
args = parser.parse_known_args()[0]



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
model.eval()
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

base_path = "/mnt/sda/feilongtang/Hallucination/SID/METHOD_EXPERIMENTS/LLAVA-1.5/OPERA/GPT4V"
if not os.path.exists(base_path + f"/{args.model}"):
    os.mkdir(base_path + f"/{args.model}")

gpt_answer_records = {}
assistant_answer_records = {}
avg_hal_score_1 = 0
avg_hal_score_2 = 0
avg_det_score_1 = 0
avg_det_score_2 = 0
num_count = 0

for idx in range(500):
    img = img_files[idx]
    image_path = args.data_path + img
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    qu = "Please describe this image in detail."
    if args.use_cd:
        image_cd = image.to(device)
    elif args.use_vcd:
        image_cd = add_diffusion_noise(image, args.noise_step)
        image_cd = image_cd.to(device)
    else:
        image_cd = None
    
    if args.use_icd:
        text_cd = 'You are a confused object detector.'
        if args.model == 'shikra':
            prompt_cd = qu[0].split("<im_end>")[0] + "<im_end>" + ' ' + text_cd + qu[0].split("<im_end>")[-1]
        elif args.model == 'llava-1.5' or args.model == 'instructblip':
            prompt_cd = qu[0].split("<ImageHere>")[0] + "<ImageHere>" + ' ' + text_cd + qu[0].split("<ImageHere>")[-1]
        # elif args.model == 'lrv_instruct' or args.model == 'minigpt4':
        else:
            prompt_cd = qu[0].split("</Img>")[0] + "</Img>" + ' ' + text_cd + qu[0].split("</Img>")[-1]

    else:
        text_cd = None    

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)
    assistant_answer_records[str(img)] = {}

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                    prompt = qu,
                    image = image.half(),
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
            )
    model_response_1 = out[0]
    assistant_answer_records[str(img)]["assistant_1"] = model_response_1
    print("Model_a output:") 
    print(model_response_1)


    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                    prompt = qu,
                    image = image.half(),
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
            )
    model_response_2 = out[0]
    assistant_answer_records[str(img)]["assistant_2"] = model_response_2
    print("Model_b output:")
    print(model_response_2)

    # gpt-4v eval
    prompt = GPT_JUDGE_PROMPT.format(model_response_1, model_response_2)

    gpt_answer = get_gpt4v_answer(prompt, image_path)
    print(gpt_answer)
    gpt_answer_records[str(img)] = gpt_answer
    print(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" "))
    print(len(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")))
    try:
        hal_score_1, hal_score_2 = gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")
        det_score_1, det_score_2 = gpt_answer.split("Detailedness: ")[-1].split("\n")[0].split(" ")

        hal_score_1 = re.match(r'\d+', hal_score_1).group()
        hal_score_2 = re.match(r'\d+', hal_score_2).group()
        det_score_1 = re.match(r'\d+', det_score_1).group()
        det_score_2 = re.match(r'\d+', det_score_2).group()
    except:
        continue
    avg_hal_score_1 += int(hal_score_1)
    avg_hal_score_2 += int(hal_score_2)
    avg_det_score_1 += int(det_score_1)
    avg_det_score_2 += int(det_score_2)
    num_count += 1
    print("=========================================")

    # dump metric file
    with open(os.path.join(base_path + f"/{args.model}", 'answers.json'), "w") as f:
        json.dump(assistant_answer_records, f)

    # dump metric file
    with open(os.path.join(base_path + f"/{args.model}", 'records.json'), "w") as f:
        json.dump(gpt_answer_records, f)

avg_score = float(avg_hal_score_1) / num_count
avg_score = float(avg_hal_score_2) / num_count
avg_score = float(avg_det_score_1) / num_count
avg_score = float(avg_det_score_2) / num_count
print(f"The avg hal score for Assistant 1 and Assistent 2: {avg_hal_score_1}; {avg_hal_score_2}")
print(f"The avg det score for Assistant 1 and Assistent 2: {avg_det_score_1}; {avg_det_score_2}")