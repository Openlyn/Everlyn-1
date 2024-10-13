#-*- encoding:utf-8 -*-
from pytorch_lightning.callbacks import Callback

class DatasetCallback(Callback):
    def __init__(self):
        self.sampler_pos_start = 0
        self.preload_used_idx_flag = False

    def on_train_start(self, trainer, pl_module):
        if not self.preload_used_idx_flag:
            self.preload_used_idx_flag = True
            trainer.train_dataloader.batch_sampler.sampler_pos_reload = self.sampler_pos_start

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.train_dataloader is not None:
            # Save sampler_pos_start parameters in the checkpoint
            checkpoint['sampler_pos_start'] = trainer.train_dataloader.batch_sampler.sampler_pos_start

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Restore sampler_pos_start parameters from the checkpoint
        if 'sampler_pos_start' in checkpoint:
            self.sampler_pos_start = checkpoint.get('sampler_pos_start', 0)
            print('Load sampler_pos_start from checkpoint, sampler_pos_start = %d' % self.sampler_pos_start)
        else:
            print('The sampler_pos_start is not in checkpoint')