import torch
import torch.nn as nn

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class Labelator(AbstractEncoder):
    """Net2Net Interface for Class-Conditional Model"""
    def __init__(self, n_classes, quantize_interface=True):
        super().__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c, **kwargs):
        c = c[:,None]
        if self.quantize_interface:
            return c, c.long()
        return c


class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x, **kwargs):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, c
        return c


class Identity(AbstractEncoder):
    # for text-conditional training
    def __init__(self, quantize_interface=True):
        super().__init__()
        self.quantize_interface = quantize_interface

    def encode(self, x, **kwargs):
        if self.quantize_interface:
            return x, x
        return x
