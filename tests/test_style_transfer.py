import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from latent_projecting import Latents, LatentPaths
from latent_projecting.style_transfer import StyleTransferer
from tests.test_projector import ProjectorTests

original_load = torch.load


def load_patch_stylegan_1(file_name, *args, **kwargs):
    file_name = str(file_name)
    if 'content' in file_name or 'style' in file_name:
        return {'latent': torch.randn(1, 14, 512), 'noise': [torch.randn((1, 1, 2**(i + 2))) for i in range(7)]}
    else:
        return original_load(file_name)


def load_patch_stylegan_2(file_name, *args, **kwargs):
    file_name = str(file_name)
    if 'content' in file_name or 'style' in file_name:
        shape_filler = lambda i: 2**(max(2, i - (i % 2) + 1))
        return {'latent': torch.randn(1, 14, 512), 'noise': [torch.randn((1, 1, shape_filler(i), shape_filler(i))) for i in range(1, 14)]}
    else:
        return original_load(file_name)


class TestStyleTransfer(ProjectorTests):

    @pytest.fixture()
    def transferer(self, device):
        self.args.device = device
        return StyleTransferer(self.args)

    def test_embed_image(self, transferer, device, stylegan_variant):
        image = Image.new("RGB", (256, 256), "black")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            image_name = temp_dir / "image.png"
            image.save(image_name)

            latents = transferer.embed_image(image_name)

        assert isinstance(latents, Latents)

        assert latents.latent.shape == (1, 14, 512)
        if stylegan_variant == "1":
            assert len(latents.noise) == 7
        else:
            assert len(latents.noise) == 13

    @pytest.mark.parametrize('style_image', [True, False])
    @pytest.mark.parametrize('content_image', [True, False])
    @patch('latent_projecting.style_transfer.torch.load')
    def test_get_latents(self, load, transferer, device, content_image, style_image, stylegan_variant):
        if stylegan_variant == '1':
            load.side_effect = load_patch_stylegan_1
        else:
            load.side_effect = load_patch_stylegan_2

        blank_image = Image.new("RGB", (256, 256), 'black')

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            if content_image:
                content_name = temp_dir / 'content.png'
                blank_image.save(content_name)
            else:
                content_name = temp_dir / 'content.pth'

            if style_image:
                style_name = temp_dir / 'style.png'
                blank_image.save(style_name)
            else:
                style_name = temp_dir / 'style.pth'

            content_latent, style_latent = transferer.get_latents(content_name, style_name)

        assert isinstance(content_latent, Latents)
        assert isinstance(style_latent, Latents)

        for latent in [content_latent, style_latent]:
            assert latent.latent.shape == (1, 14, 512)
            if stylegan_variant == '1':
                assert len(latent.noise) == 7
            else:
                assert len(latent.noise) == 13

            assert device in str(latent.latent.device)
            for noise in latent.noise:
                assert device in str(noise.device)

    def test_post_noise_optimize(self, transferer):
        content_latent = transferer.projector.create_initial_latent_and_noise()
        transfer_latent = transferer.projector.create_initial_latent_and_noise()

        result = transferer.post_noise_optimize(content_latent, transfer_latent)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], LatentPaths)
        assert isinstance(result[1], Latents)

    @pytest.mark.parametrize('post_optimize', [True, False])
    def test_do_style_transfer(self, post_optimize, device):
        self.args.device = device
        self.args.post_optimize = post_optimize
        transferer = StyleTransferer(self.args)

        content_latent = transferer.projector.create_initial_latent_and_noise()
        transfer_latent = transferer.projector.create_initial_latent_and_noise()

        results = [transferer.do_style_transfer(content_latent, transfer_latent, i) for i in range(14)]

        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], torch.Tensor)
            if post_optimize:
                assert isinstance(result[1], LatentPaths)
            else:
                assert result[1] is None
