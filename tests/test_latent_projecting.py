import pytest
import torch

from latent_projecting import noise_loss, w_plus_style_loss, naive_noise_loss, w_plus_loss

possible_devices = ['cpu']
if torch.cuda.is_available():
    possible_devices += ['cuda']


class TestLosses:

    def run(self, loss_func, device='cpu'):
        losses = loss_func(
            torch.randn(self.shape).to(device),
            torch.randn(self.shape).to(device)
        )

        assert isinstance(losses, tuple)
        assert len(losses) == 2
        assert isinstance(losses[0], torch.Tensor)
        assert isinstance(losses[1], dict)

    @pytest.fixture(autouse=True)
    def shape(self):
        self.shape = 1, 3, 256, 256
        self.mask_shape = 1, 1, 256, 256

    def test_noise_loss(self):
        lambdas = {"l_mse_1": 1, "l_mse_2": 1}
        loss_func = noise_loss(lambdas, torch.randn(self.shape), torch.randn(self.shape), torch.randn(self.mask_shape))
        self.run(loss_func)

    @pytest.mark.parametrize("device", possible_devices)
    def test_w_pluss_style_loss(self, device):
        lambdas = {"l_style": 1, "l_percept": 1, "l_mse": 1}
        loss_func = w_plus_style_loss(
            lambdas,
            torch.randn(self.shape).to(device),
            torch.randn(self.shape).to(device),
            torch.randn(self.mask_shape).to(device),
            device
        )
        self.run(loss_func, device=device)

    def test_naive_noise_loss(self):
        lambdas = {"l_mse": 1}
        loss_func = naive_noise_loss(lambdas)
        self.run(loss_func)

    @pytest.mark.parametrize("device", possible_devices)
    def test_w_plus_loss(self, device):
        lambdas = {"l_percept": 1, "l_mse": 1}
        loss_func = w_plus_loss(lambdas, device)
        self.run(loss_func, device=device)
