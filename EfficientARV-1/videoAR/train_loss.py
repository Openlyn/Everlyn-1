
import torch
import torch.nn.functional as F
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class TrainLoss:
    def __init__(self, loss_type='sample_mse'):
        self.loss_type = loss_type
    
    def train_losses(self, model, batch, target=None):
        

        pred_token = model(batch)
        
        loss_dict = {}
        if self.loss_type == 'dist':
            latent_dist = target['latent_dist']
            loss = pred_token.kl(latent_dist)
            loss_dict['loss'] = loss
        elif self.loss_type == 'mse':
            dims = list(range(1, pred_token.mean.ndim))
            latent_dist = target['latent_dist']
            loss = torch.sum(F.mse_loss(pred_token.mean.float(), latent_dist.mean.float(), reduction='none'), dim=dims) + \
                    torch.sum(F.mse_loss(pred_token.logvar.float(), latent_dist.logvar.float(), reduction='none'), dim=dims)
            loss_dict['loss'] = loss
        elif self.loss_type == 'sample_mse':
            latents = target['latents']
            b, c, t, h, w = latents.shape
            loss = (pred_token.sample()-latents)**2
            loss = mean_flat(loss)
            loss_dict['loss'] = loss
        elif self.loss_type == 'cross_entropy':
            loss_dict['loss'] = pred_token
        elif self.loss_type == 'ce_mse':
            loss_dict['loss'] = pred_token[0]
            loss_dict['ce_loss'] = pred_token[1]
            loss_dict['feature_loss'] = pred_token[2]
        else:
            raise NotImplementedError
        return loss_dict
        