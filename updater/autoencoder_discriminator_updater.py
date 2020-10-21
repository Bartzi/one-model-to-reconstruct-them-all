import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import UpdateDisabler, GradientApplier

from updater.autoencoder_updater import AutoencoderUpdater


class AutoencoderDiscriminatorUpdater(AutoencoderUpdater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularization_settings = {
            "d_interval": 16,
            "r1_weight": 10,
        }

    def get_discriminator(self) -> nn.Module:
        discriminator = self.networks['discriminator']
        if isinstance(discriminator, DistributedDataParallel):
            discriminator_module = discriminator.module
        else:
            discriminator_module = discriminator
        return discriminator_module

    def update_core(self):
        reporter = get_current_reporter()
        image_batch = next(self.iterators['images'])
        image_batch = {k: v.to(self.device) for k, v in image_batch.items()}

        discriminator_observations = self.update_discriminator(
            image_batch['input_image'].clone().detach(),
            image_batch['output_image'].clone().detach(),
        )
        reporter.add_observation(discriminator_observations, 'discriminator')

        generator_observations = self.update_generator(
            image_batch['input_image'].clone().detach(),
            image_batch['output_image'].clone().detach(),
        )
        reporter.add_observation(generator_observations, 'generator')

    def update_discriminator(self, input_images: torch.Tensor, output_images: torch.Tensor) -> dict:
        autoencoder = self.get_autoencoder()
        discriminator = self.get_discriminator()
        discriminator_optimizer = self.optimizers['discriminator']

        with UpdateDisabler(autoencoder), GradientApplier([discriminator], [discriminator_optimizer]):
            reconstructed_image = autoencoder(input_images)
            fake_prediction = discriminator(reconstructed_image)
            fake_loss = F.softplus(fake_prediction).mean()
            fake_loss.backward()

            real_prediction = discriminator(output_images.detach())
            real_loss = F.softplus(-real_prediction).mean()
            real_loss.backward()

            discriminator_loss = real_loss.detach() + fake_loss.detach()

        loss_data = {
            'loss': discriminator_loss,
            'real_score': real_prediction.mean(),
            'fake_score': fake_prediction.mean()
        }

        if self.iteration % self.regularization_settings['d_interval'] == 0:
            image.requires_grad = True
            real_prediction = discriminator(image)
            grad_of_reference_image, = torch.autograd.grad(outputs=real_prediction.sum(), inputs=image, create_graph=True)
            gradient_penalty = grad_of_reference_image.pow(2).view(grad_of_reference_image.shape[0], -1).sum(1).mean()

            discriminator.zero_grad()
            (self.regularization_settings['r1_weight'] / 2 * gradient_penalty * self.regularization_settings['d_interval'] + 0 * real_prediction[0]).backward()
            discriminator_optimizer.step()

            loss_data['gradient_penalty'] = self.regularization_settings['r1_weight'] / 2 * gradient_penalty.detach().cpu() * self.regularization_settings['d_interval']

        torch.cuda.empty_cache()

        return loss_data

    def update_generator(self, input_images: torch.Tensor, output_images: torch.Tensor) -> dict:
        autoencoder = self.get_autoencoder()
        discriminator = self.get_discriminator()

        reporter = get_current_reporter()

        autoencoder_optimizer = self.optimizers['main']
        log_data = {}

        with UpdateDisabler(autoencoder.decoder), GradientApplier([autoencoder], [autoencoder_optimizer]):
            reconstructed_images = autoencoder(input_images)

            mse_loss = F.mse_loss(output_images, reconstructed_images, reduction='none')
            loss = mse_loss.mean(dim=(1, 2, 3)).sum()
            reporter.add_observation({"reconstruction_loss": loss}, prefix='loss')
            if self.use_perceptual_loss:
                perceptual_loss = self.perceptual_loss(reconstructed_images, output_images).sum()
                loss += perceptual_loss
                reporter.add_observation(
                    {"autoencoder_loss": loss, "perceptual_loss": perceptual_loss},
                    prefix='loss'
                )

            discriminator_prediction = discriminator(reconstructed_images)
            discriminator_loss = F.softplus(-discriminator_prediction).mean()

            loss += discriminator_loss
            loss.backward()

        log_data.update({
            "loss": loss,
            "discriminator_loss": discriminator_loss,
        })
        torch.cuda.empty_cache()

        return log_data

