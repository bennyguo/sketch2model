import math

import torch
from torch.optim import lr_scheduler


def init_net(net, opt):
    if opt.n_gpus > 0:
        assert(torch.cuda.is_available())
        net.to(opt.device)
        net = torch.nn.DataParallel(net)  # multi-GPUs
    return net


def get_scheduler(optimizer, opt, last_epoch=-1):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | exp | step
    """
    if opt.lr_policy == 'linear':
        """Linear decay in the last opt.n_epochs_decay epochs."""
        def lambda_rule(epoch):
            t = max(0, epoch + 1 - opt.n_epochs + opt.n_epochs_decay) / float(opt.n_epochs_decay + 1)
            lr = opt.lr * (1 - t) + opt.lr_final * t
            return lr / opt.lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'exp':
        """Exponential decay in the last opt.n_epochs_decay epochs."""
        def lambda_rule(epoch):
            t = max(0, epoch + 1 - opt.n_epochs + opt.n_epochs_decay) / float(opt.n_epochs_decay + 1)
            lr = math.exp(math.log(opt.lr) * (1 - t) + math.log(opt.lr_final) * t)
            return lr / opt.lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        """Decay every opt.lr_decay_epochs epochs."""
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_gamma, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler    
