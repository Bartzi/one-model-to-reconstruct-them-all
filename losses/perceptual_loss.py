from torch import nn, Tensor
from torch.nn import functional as F


class PerceptualLoss(nn.Module):

    def __init__(self, target: Tensor, mask: Tensor = None):
        super().__init__()
        if mask is not None:
            target = mask * target
        self.target = target.detach()
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        if self.mask is not None:
            x = x * self.mask.detach()

        loss = F.mse_loss(x, self.target)
        return loss
