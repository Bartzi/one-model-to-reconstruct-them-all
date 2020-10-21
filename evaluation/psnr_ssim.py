import torch
from kornia import psnr_loss, ssim as ssim_loss
from typing import Tuple

from pytorch_training.images.utils import clamp_and_unnormalize


class PSNRSSIMEvaluator:

    def __init__(self, max_value: int = 1, ssim_kernel_size: int = 5):
        self.max_value = max_value
        self.ssim_kernel_size = ssim_kernel_size

    def unnormalize(self, image: torch.Tensor) -> torch.Tensor:
        if image.min() < 0:
            image = clamp_and_unnormalize(image)
        return image

    def psnr(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        image = self.unnormalize(image)
        target = self.unnormalize(target)

        assert len(image) == 1, "Batch size of images must be one in order to get a meaningful psnr result"
        psnr = psnr_loss(image, target, self.max_value)
        return psnr

    def ssim(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        image = self.unnormalize(image)
        target = self.unnormalize(target)

        assert len(image) == 1, "Batch size of images must be one in order to get a meaningful ssim result"
        ssim = ssim_loss(image, target, self.ssim_kernel_size, reduction='none')
        ssim = (1 - 2 * ssim).mean()
        return ssim

    def psnr_and_ssim(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        psnr = self.psnr(image, target)
        ssim = self.ssim(image, target)
        return psnr, ssim
