import torch
from torch import nn

from losses import gram_matrix


class StyleLoss(nn.Module):

    def __init__(self, target_feature: torch.Tensor, mask=None):
        super().__init__()
        self.target_gram_matrix = gram_matrix(target_feature, mask).detach()
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels = x.shape[:2]

        G = gram_matrix(x, self.mask)
        loss = (G - self.target_gram_matrix).square().sum() / (4 * (batch_size * num_channels)**2)

        return loss
