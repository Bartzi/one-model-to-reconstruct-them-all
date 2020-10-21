import copy

import pytest
import torch

from latent_projecting import LatentPaths, Latents
from tests.test_projector import possible_devices


class TestLatentPaths:

    @pytest.fixture
    def path_length(self):
        return 20

    @pytest.fixture
    def path(self, path_length):
        latent = [torch.ones((1, 14, 512)) for _ in range(path_length)]
        noises = [[torch.ones((1, 1, 4, 4)) for _ in range(7)] for __ in range(path_length)]

        path = LatentPaths(latent, noises)
        return path

    def test_len(self, path, path_length):
        assert len(path) == path_length

        path = LatentPaths(path.latent, path.noise[:-2])

        with pytest.raises(AssertionError):
            assert len(path) == path_length

    @pytest.mark.parametrize('device', possible_devices)
    def test_to(self, device, path):
        path = path.to(device)

        for element in path:
            assert device in str(element.latent.device)
            for noise in element.noise:
                assert device in str(noise.device)

    def test_add(self, path, path_length):
        second_path = copy.deepcopy(path)
        path = path + second_path

        assert len(path.latent) == 2 * path_length
        assert len(path.noise) == 2 * path_length

    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    def test_split(self, path_length, batch_size):
        latent = [torch.ones((batch_size, 14, 512)) for _ in range(path_length)]
        noises = [[torch.ones((batch_size, 1, 4, 4)) for _ in range(7)] for __ in range(path_length)]
        path = LatentPaths(latent, noises)

        splitted_path = path.split()

        assert len(splitted_path) == batch_size
        for split in splitted_path:
            assert len(split) == path_length
            for element in split:
                assert element.latent.shape == (1, 14, 512)
                for noise in element.noise:
                    assert noise.shape == (1, 1, 4, 4)


class TestLatents:

    @pytest.mark.parametrize('device', possible_devices)
    def test_to(self, device):
        def check_device(latents, dev):
            assert dev in str(latents.latent.device)
            for noise in latents.noise:
                assert dev in str(noise.device)

        latent = torch.ones((1, 14, 512))
        noises = [torch.ones((1, 1, 4, 4)) for _ in range(7)]

        latents = Latents(latent, noises)
        check_device(latents, 'cpu')

        latents = latents.to(device)
        check_device(latents, device)

    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    def test_getitem(self, batch_size):
        latent = torch.ones((batch_size, 14, 512))
        noises = [torch.ones((batch_size, 1, 4, 4)) for _ in range(7)]

        latents = Latents(latent, noises)

        for i in range(batch_size):
            sub_latent = latents[i]
            assert sub_latent.latent.shape == (1, 14, 512)
            for noise in sub_latent.noise:
                assert noise.shape == (1, 1, 4, 4)
