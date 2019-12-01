from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class CameraInvariantLoss(nn.Module):
    """
    CCSC LOSS

    """


    def __init__(self, metric='dot'):
        super(CameraInvariantLoss, self).__init__()
        self.metric = metric

    def forward(self, inputs, pids, camids):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        if self.metric == 'eculid':
            # Compute pairwise distance, replace by the official when merged
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            # calculate sqrt
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

            # For each anchor, find the hardest positive and negative
            mask_pid = pids.expand(n, n).eq(pids.expand(n, n).t())
            mask_camid = camids.expand(n, n).ne(camids.expand(n, n).t())
            mask = mask_pid & mask_camid

            loss = torch.mean(dist[mask == 1])

        elif self.metric == 'dot':
            similarity = inputs.mm(inputs.t())
            norm = torch.norm(inputs,p=2,dim=1,keepdim=True)
            norm_dist = norm.mm(norm.t())
            # normilize simmilarity to range in [-1,1]
            simi = similarity/norm_dist
            dist = 1/(1+simi)
            # For each anchor, find the hardest positive and negative
            mask_pid = pids.expand(n, n).eq(pids.expand(n, n).t())
            mask_camid = camids.expand(n, n).ne(camids.expand(n, n).t())
            mask = mask_pid & mask_camid

            loss = torch.mean(dist[mask == 1])

        elif self.metric == 'dual':

            similarity = inputs.mm(inputs.t())
            norm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            norm_dist = norm.mm(norm.t())

            # normilize simmilarity to range in [-1,1]
            simi = similarity / norm_dist
            dist = 1 / (1 + simi)

            # For each anchor, find the hardest positive and negative
            mask_pid_max = pids.expand(n, n).eq(pids.expand(n, n).t())
            mask_camid_max = camids.expand(n, n).ne(camids.expand(n, n).t())
            mask_max = mask_pid_max & mask_camid_max

            mask_pid_min = pids.expand(n, n).ne(pids.expand(n, n).t())
            mask_camid_min = camids.expand(n, n).eq(camids.expand(n, n).t())
            mask_min = mask_pid_min & mask_camid_min

            loss_simi_max = torch.mean(dist[mask_max == 1])
            loss_simi_min = torch.mean(simi[mask_min == 1])

            # loss_simi_max [0.7,0.5], loss_simi_min [0.4,0.2]
            loss = 1.0 * loss_simi_max + 1.0 * loss_simi_min
        else:
            pass

        return loss












