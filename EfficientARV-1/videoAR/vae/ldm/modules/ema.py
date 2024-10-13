#-*- encoding:utf-8 -*-
import torch
from torch import nn
from pytorch_lightning.callbacks import Callback

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

class EMACallback(Callback):
    def __init__(self, decay=0.9999):
        self.decay = decay
        self.shadow_params = {}

    def on_train_start(self, trainer, pl_module):
        # initialize shadow parameters for original models
        total_ema_cnt = 0
        for name, param in pl_module.named_parameters():
            if name not in self.shadow_params:
                self.shadow_params[name] = param.data.clone()
            else: # already in dict, maybe load from checkpoint
                pass
            print('will calc ema for param: %s' % name)
            total_ema_cnt += 1
        print('total_ema_cnt=%d' % total_ema_cnt)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update the shadow params at the end of each epoch
        for name, param in pl_module.named_parameters():
            assert name in self.shadow_params
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow_params[name]
            self.shadow_params[name] = new_average.clone()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save EMA parameters in the checkpoint
        checkpoint['ema_params'] = self.shadow_params

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Restore EMA parameters from the checkpoint
        if 'ema_params' in checkpoint:
          self.shadow_params = checkpoint.get('ema_params', {})
          for k in self.shadow_params:
                self.shadow_params[k] = self.shadow_params[k].cuda()
          print('load shadow params from checkpoint, cnt=%d' % len(self.shadow_params))
        else:
          print('ema_params is not in checkpoint')