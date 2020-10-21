import pytest
import torch
from torch import optim

from latent_projecting import naive_noise_loss, optimize_noise, LatentPaths, Latents, run_image_reconstruction, \
    run_local_style_transfer
from tests.test_projector import ProjectorTests, possible_devices


class TestProjectFunctions(ProjectorTests):

    def get_input_data(self, projector, device):
        latents = projector.create_initial_latent_and_noise().to(device)
        images = torch.ones(self.shape).to(device)
        lambdas = {"l_mse": 1}
        loss_func = naive_noise_loss(lambdas)
        return latents, images, loss_func

    def test_optimize_noise(self, projector, device):
        input_data = self.get_input_data(projector, device)
        result = optimize_noise(self.args, projector, *input_data)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], LatentPaths)
        assert isinstance(result[1], Latents)

    @pytest.mark.parametrize("latent_abort_condition", [None, lambda loss_dict: loss_dict['psnr'] < 100])
    @pytest.mark.parametrize("noise_abort_condition", [None, lambda loss_dict: loss_dict['psnr'] < 100])
    @pytest.mark.parametrize("do_optimize_noise", [True, False])
    def test_run_image_reconstruction(self, projector, device, do_optimize_noise, latent_abort_condition, noise_abort_condition):
        latents, images, _ = self.get_input_data(projector, device)
        result = run_image_reconstruction(
            self.args,
            projector,
            latents,
            images,
            do_optimize_noise=do_optimize_noise,
            latent_abort_condition=latent_abort_condition,
            noise_abort_condition=noise_abort_condition
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], LatentPaths)
        assert isinstance(result[1], Latents)

    def test_run_local_style_transfer(self, projector, device):
        latents = projector.create_initial_latent_and_noise().to(device)
        mask_shape = (1, 1,) + self.shape[-2:]

        self.args.style_latent_step = 5
        self.args.style_lr_rampdown = 0
        self.args.style_lr_rampup = 0

        self.args.style_noise_step = 5
        self.args.noise_style_lr_rampdown = 0
        self.args.noise_style_lr_rampup = 0

        result = run_local_style_transfer(
            self.args,
            projector,
            latents,
            torch.randn(self.shape).to(device),
            torch.randn(self.shape).to(device),
            torch.randn(mask_shape).to(device),
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], LatentPaths)
        assert isinstance(result[1], Latents)
