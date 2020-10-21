import torch
import torch.nn.functional as F
from kornia import ssim as ssim_loss, psnr_loss

from losses.lpips import PerceptualLoss
from networks import StyleganAutoencoder
from pytorch_training.images.utils import clamp_and_unnormalize
from pytorch_training.reporter import get_current_reporter


class AutoEncoderEvalFunc:

    def __init__(self, autoencoder: StyleganAutoencoder, device: int, use_perceptual_loss: bool = True):
        self.autoencoder = autoencoder
        self.perceptual_loss = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[device])
        self.use_perceptual_loss = use_perceptual_loss

    def __call__(self, batch):
        reporter = get_current_reporter()

        with torch.no_grad():
            reconstructed_images = self.autoencoder(batch['input_image'])
            original_image = batch['output_image']

            mse_loss = F.mse_loss(original_image, reconstructed_images, reduction='none')
            loss = mse_loss.mean(dim=(1, 2, 3)).sum()
            reporter.add_observation({"reconstruction_loss": loss}, prefix='evaluation')
            if self.use_perceptual_loss:
                perceptual_loss = self.perceptual_loss(reconstructed_images, original_image).sum()
                reporter.add_observation({"perceptual_loss": perceptual_loss}, prefix='evaluation')
                loss += perceptual_loss

            original_image = clamp_and_unnormalize(original_image)
            reconstructed_images = clamp_and_unnormalize(reconstructed_images)
            psnr = psnr_loss(reconstructed_images, original_image, max_val=1)

            ssim = ssim_loss(original_image, reconstructed_images, 5, reduction='mean')
            # since we get a loss, we need to calculate/reconstruct the original ssim value
            ssim = 1 - 2 * ssim

            reporter.add_observation({"psnr": psnr, "ssim": ssim}, prefix='evaluation')

        reporter.add_observation({"autoencoder_loss": loss}, prefix='evaluation')
