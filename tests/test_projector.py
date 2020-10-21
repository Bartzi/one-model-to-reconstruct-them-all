import argparse
import tempfile
from pathlib import Path
from shutil import which

import numpy as np
import pytest
import torch
from PIL import Image
from torch import optim

from latent_projecting import Latents, naive_noise_loss, LatentPaths
from latent_projecting.projector import Projector
from pytorch_training.data import Compose
from utils.command_line_args import add_default_args_for_projecting


possible_devices = ['cpu', 'cuda']


@pytest.mark.parametrize("stylegan_variant", ["1", "2"])
class ProjectorTests:

    @pytest.fixture(autouse=True)
    def set_up(self, stylegan_variant):
        parser = argparse.ArgumentParser()
        parser = add_default_args_for_projecting(parser)

        self.args = parser.parse_args(args=[])
        self.args.latent_step = 5
        self.args.noise_step = 5
        self.shape = 1, 3, 256, 256
        self.args.config = Path(__file__).parent / "testdata" / f"config_stylegan_{stylegan_variant}.json"

    @pytest.fixture(params=possible_devices)
    def device(self, request):
        if request.param == 'cuda' and not torch.cuda.is_available():
            pytest.skip("can not run cuda tests, since no GPU is available")
        return request.param

    @pytest.fixture(autouse=True)
    def maybe_skip(self, stylegan_variant, device):
        if stylegan_variant == '2' and device == 'cpu':
            pytest.skip("Stylegan2 does not work on CPU!")

    @pytest.fixture
    def projector(self, stylegan_variant, device):
        self.args.device = device
        return Projector(self.args)


class TestProjector(ProjectorTests):

    def test_config(self, stylegan_variant, projector):
        config = projector.config
        assert config['stylegan_variant'] == int(stylegan_variant)
        for key in ['image_size', 'latent_size', 'stylegan_checkpoint']:
            assert config.get(key, None) is not None

    def test_config_without_input(self, stylegan_variant):
        self.args.config = None
        with pytest.raises(RuntimeError):
            Projector(self.args)

    def test_get_blur_transform(self, stylegan_variant, projector):
        transform = projector.get_blur_transform()
        assert len(transform) == 3

        transform = projector.get_blur_transform(from_tensor=False)
        assert len(transform) == 1

    def test_get_transforms(self, projector):
        transform = projector.get_transforms()
        assert len(transform.transforms) == 3
        assert isinstance(transform, Compose)

        image = Image.new('RGB', (512, 512), 'black')
        transformed = transform(image)

        assert transformed.min() == -1
        assert transformed.shape == (3, projector.config['image_size'], projector.config['image_size'])

        image = Image.new('RGB', (128, 128), 'white')
        transformed = transform(image)

        assert transformed.max() == 1
        assert transformed.shape == (3, projector.config['image_size'], projector.config['image_size'])

    def test_get_mask_transform(self, projector):
        def transform_image(max_val, transformation):
            image = np.random.random((224, 224, 3))
            image[0:20, 0:20, :] = max_val
            image[20:40, 20:40, :] = 0
            image = (image * 255).astype('uint8')
            image = Image.fromarray(image, 'RGB')
            return transformation(image)

        transform = projector.get_mask_transform()
        assert len(transform.transforms) == 6
        transformed = transform_image(1, transform)

        assert transformed.min() == 0
        assert transformed.max() == 1
        assert transformed.shape == (1, projector.config['image_size'], projector.config['image_size'])

        transform = projector.get_mask_transform(invert_mask=True)
        assert len(transform.transforms) == 7
        transformed = transform_image(1, transform)

        assert transformed[0, 0, 0] == 0
        assert transformed[0, 30, 30] == 1
        assert transformed.shape == (1, projector.config['image_size'], projector.config['image_size'])

        transform = projector.get_mask_transform(mask_multiplier=0.9)
        assert len(transform.transforms) == 7
        transformed = transform_image(0.9, transform)

        assert transformed[0, 0, 0] < 1
        assert transformed[0, 30, 30] == 0
        assert transformed.min() == 0
        assert transformed.shape == (1, projector.config['image_size'], projector.config['image_size'])

    def test_get_mean_latent(self, projector):
        mean_latent = projector.get_mean_latent(1000)

        assert isinstance(mean_latent, tuple)
        assert len(mean_latent) == 2

        assert mean_latent[0].numel() == projector.config['latent_size']
        assert mean_latent[1].numel() == 1

    def test_requires_grad(self, projector):
        latents = projector.create_initial_latent_and_noise()
        assert latents.latent.requires_grad is False
        for noise in latents.noise:
            assert noise.requires_grad is False

        projector.set_requires_grad(latents, True)
        assert latents.latent.requires_grad is True
        for noise in latents.noise:
            assert noise.requires_grad is False

        projector.set_requires_grad(latents, False)
        assert latents.latent.requires_grad is False
        for noise in latents.noise:
            assert noise.requires_grad is True

    def test_generate(self, projector, device):
        latents = projector.create_initial_latent_and_noise()
        generated = projector.generate(latents)[0]

        assert isinstance(generated, torch.Tensor)
        assert generated.shape[-2:] == (projector.config['image_size'], projector.config['image_size'])

    def run_project(self, projector, device):
        latents = projector.create_initial_latent_and_noise().to(device)
        images = torch.ones(self.shape).to(device)
        optimizer = optim.Adam([latents.latent], lr=projector.args.lr)
        lambdas = {"l_mse": 1}
        loss_func = naive_noise_loss(lambdas)

        return projector.project(latents, images, optimizer, 5, loss_func)

    def test_project(self, projector, device):
        result = self.run_project(projector, device)
        assert isinstance(result, tuple)
        assert len(result) == 2

        assert isinstance(result[0], LatentPaths)
        assert isinstance(result[1], Latents)

        assert result[1].latent.shape[2] == projector.config['latent_size']

    @pytest.mark.skipif(which('convert') is None, reason="Convert not installed on system and necessary for gif creation")
    def test_create_gif(self, projector, device):
        latent_paths, _ = self.run_project(projector, device)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            file_name = 'test'
            projector.create_gif(latent_paths, file_name, temp_dir)
            assert (temp_dir / 'gifs' / f"{file_name}.gif").exists()

    def test_render_log(self, projector, device):
        latent_paths, _ = self.run_project(projector, device)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            base_name = 'test'
            projector.render_log(temp_dir, base_name)

            assert (temp_dir / 'log' / f"{base_name}_log.json").exists()

            possible_keys = projector.log[0].keys()
            for key in possible_keys:
                assert (temp_dir / 'log' / f"{base_name}_{key}.png").exists()
