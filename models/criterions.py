import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import soft_renderer as sr


class LaplacianLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.template_mesh = sr.Mesh.from_obj(opt.template_path)
        self.loss = sr.LaplacianLoss(
            self.template_mesh.vertices[0].cpu(),
            self.template_mesh.faces[0].cpu()
        ).to(opt.device)

    def forward(self, v):
        return self.loss(v).mean()


class FlattenLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.template_mesh = sr.Mesh.from_obj(opt.template_path)
        self.loss = sr.FlattenLoss(
            self.template_mesh.faces[0].cpu()
        ).to(opt.device)
    
    def forward(self, v):
        return self.loss(v).mean()


def iou(pred, target, eps=1e-6):
    dims = tuple(range(pred.ndimension())[1:])
    intersect = (pred * target).sum(dims)
    union = (pred + target - pred * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()


class IoULoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, pred, target):
        return 1 - iou(pred[:, 3], target[:, 3])


class MultiViewIoULoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, pred, target_a, target_b):
        return (
            1 - iou(pred[0][:, 3], target_a[:, 3]) +
            1 - iou(pred[1][:, 3], target_a[:, 3]) +
            1 - iou(pred[2][:, 3], target_b[:, 3]) +
            1 - iou(pred[3][:, 3], target_b[:, 3])
        ) / 4.


class MSELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, pred, target):
        return F.mse_loss(pred, target, reduction='mean')


def voxel_iou(pred, target):
    return ((pred * target).sum((1, 2, 3)) / (0 < (pred + target)).sum((1, 2, 3))).mean()
