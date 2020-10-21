import cv2
import numpy as np
import pytest
import torch

from losses.perceptual_loss import PerceptualLoss
from losses.perceptual_style_loss import FixedPerceptualAndStyleLoss
from losses.psnr import PSNR
from losses.style_loss import StyleLoss


class TestPSNR:

    @pytest.fixture(autouse=True)
    def psnr(self):
        self.psnr_func = PSNR()

    def test_psnr_same_input(self):
        image = torch.ones((5, 5))

        psnr = self.psnr_func(image, image)
        assert float(psnr) == float('inf')

    def test_psnr_random_input(self):
        image_1 = np.random.random((5, 5))
        image_2 = np.random.random((5, 5))

        cv2_psnr = cv2.PSNR(image_1, image_2, R=1)
        psnr = self.psnr_func(torch.Tensor(image_1), torch.Tensor(image_2))

        assert float(cv2_psnr) == pytest.approx(float(psnr))


class TestStyleLoss:

    @pytest.fixture(autouse=True)
    def shape(self):
        self.shape = 1, 3, 256, 256

    def test_return_type_style(self):
        loss_func = StyleLoss(torch.ones(self.shape))
        loss = loss_func(torch.zeros(self.shape))
        assert isinstance(loss, torch.Tensor)

    def test_return_type_perceptual(self):
        loss_func = PerceptualLoss(torch.ones(self.shape))
        loss = loss_func(torch.zeros(self.shape))
        assert isinstance(loss, torch.Tensor)

    def test_return_type_combined(self):
        loss_func = FixedPerceptualAndStyleLoss(torch.ones(self.shape), torch.ones(self.shape))
        losses = loss_func(torch.zeros(self.shape))
        assert isinstance(losses, tuple)
        assert len(losses) == 2
        for loss in losses:
            assert isinstance(loss, torch.Tensor)
