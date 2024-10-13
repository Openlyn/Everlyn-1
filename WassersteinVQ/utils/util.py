import sys
import os
from collections import defaultdict
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from torch import nn, optim
import math
from omegaconf import OmegaConf
import yaml
from torch import inf

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

'''
def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.ae_lr * epoch / args.warmup_epochs 
    else:
        end_lr = args.ae_lr * 0.001
        q = 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        lr = args.ae_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
'''

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.ae_lr * epoch / args.warmup_epochs
    else:
        lr = args.ae_lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class Logger(object):
    def __init__(self, saver_dir, saver_name_pre):
        self.terminal = sys.stdout
        output_file = os.path.join(saver_dir, saver_name_pre+"-record.log") 
        self.log = open(output_file, "w")

    def write(self, message):
        print(message, end="", file=self.terminal, flush=True)
        print(message, end="", file=self.log, flush=True)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list[v]
            else:
                pack[k] = v
        return pack

class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                try:
                    self.losses[key].append(val.item())
                except:
                    self.losses[key].append(val)
                    
    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            else:
                avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
                str_losses.append("{} {:.4f},".format(key, avg_loss))
        if prefix:
            return "{} {}".format(prefix, " ".join(str_losses))
        else:
            return "{}".format(" ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses) 

