import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from losses.lpips import PerceptualLoss
from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import UpdateDisabler, GradientApplier


class AutoencoderUpdater(Updater):

    def __init__(self, *args, use_perceptual_loss: bool = True, disable_update_for: str = 'none', **kwargs):
        super().__init__(*args, **kwargs)
        self.perceptual_loss = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[self.device])
        self.use_perceptual_loss = use_perceptual_loss
        self.disable_update(disable_update_for)

    def get_autoencoder(self) -> nn.Module:
        autoencoder = self.networks['autoencoder']
        if isinstance(autoencoder, DistributedDataParallel):
            autoencoder_module = autoencoder.module
        else:
            autoencoder_module = autoencoder
        return autoencoder_module

    def disable_update(self, disable_update_for: str):
        if disable_update_for == 'none':
            return

        autoencoder = self.get_autoencoder()

        disable_noise = disable_update_for == 'noise'
        for name, parameter in autoencoder.encoder.named_parameters():
            if 'noise' in name:
                parameter.requires_grad = not disable_noise
            else:
                parameter.requires_grad = disable_noise

        if disable_noise:
            autoencoder.use_generated_noise = False

    def calculate_loss(self, input_images: torch.Tensor, reconstructed_images: torch.Tensor):
        reporter = get_current_reporter()

        mse_loss = F.mse_loss(input_images, reconstructed_images, reduction='none')
        loss = mse_loss.mean(dim=(1, 2, 3)).sum()
        reporter.add_observation({"reconstruction_loss": loss}, prefix='loss')
        if self.use_perceptual_loss:
            perceptual_loss = self.perceptual_loss(reconstructed_images, input_images).sum()
            reporter.add_observation({"perceptual_loss": perceptual_loss}, prefix='loss')
            loss += perceptual_loss

        loss.backward()
        reporter.add_observation({"autoencoder_loss": loss}, prefix='loss')

    def update_core(self):
        autoencoder = self.get_autoencoder()

        with UpdateDisabler(autoencoder.decoder), GradientApplier([autoencoder], [self.optimizers['main']]):
            image_batch = next(self.iterators['images'])
            image_batch = {k: v.to(self.device) for k, v in image_batch.items()}

            reconstructed_images = autoencoder(image_batch['input_image'])

            self.calculate_loss(image_batch['output_image'], reconstructed_images)

