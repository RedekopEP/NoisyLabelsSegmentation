"""
Here presented main Metrics
- IOU
- DICE
"""

import numpy as np
import torch


def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    eps = 1e-15
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
    return (float(intersection) + eps) / (union + eps)


def calc_iou_multi(prediction, ground_truth, num_class):
    cls = num_class
    ground_truth_cl = np.zeros(ground_truth.shape)
    ground_truth_cl[ground_truth == cls + 1] = 1
    n_images = len(prediction)
    intersection, union = 0, 0
    eps = 1e-15
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth_cl[i] > 0).astype(np.float32).sum()
        union += np.logical_or(prediction[i] > 0, ground_truth_cl[i] > 0).astype(np.float32).sum()
    iou = (float(intersection) + eps) / (union + eps)
    return iou


### calculated DICE metric
def calc_md(prediction, ground_truth):
    eps = 1e-15
    # print('started')
    intersection = np.logical_and(prediction > 0, ground_truth > 0).astype(np.float32).sum()
    # print('finished')
    return (2. * intersection.sum() + eps) / ((prediction > 0).sum() + (ground_truth > 0).sum() + eps)


### calculated DICE metric
def calc_md_multi(prediction, ground_truth, num_class):
    cls = num_class
    smooth = 1e-15
    outputs_cls = prediction[:, cls, :, :, :].contiguous().view(-1)
    if cls == 0:
        targets_cls = (ground_truth > 0.0).float().contiguous().view(-1)
    else:
        targets_cls = (ground_truth == cls + 1).float().contiguous().view(-1)
    target_cls = (targets_cls > 0.0).float()
    md = (2 * torch.sum(outputs_cls * target_cls, dim=0) + smooth) / (
            torch.sum(outputs_cls, dim=0) + torch.sum(target_cls, dim=0) + smooth)
    # md = (2. * (intersection.sum() + eps) / ((prediction > 0).sum() + (ground_truth_cl > 0).sum() + eps))
    return md.mean().detach().numpy()
