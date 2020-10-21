import os
from pathlib import Path
from typing import Union, Dict, Iterable, Type

import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from data.autoencoder_dataset import AutoencoderDataset
from latent_projecting import Latents
from networks import StyleganAutoencoder
from pytorch_training.data.utils import default_loader
from pytorch_training.distributed import get_world_size, get_rank


def resilient_loader(path):
    try:
        return default_loader(path)
    except Exception:
        return Image.new('RGB', (256, 256))


def build_data_loader(image_path: Union[str, Path], config: dict, uses_absolute_paths: bool, shuffle_off: bool = False, dataset_class: Type[AutoencoderDataset] = AutoencoderDataset) -> DataLoader:
    transform_list = [
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * config['input_dim'], (0.5,) * config['input_dim'])
    ]
    transform_list = transforms.Compose(transform_list)

    dataset = dataset_class(
        image_path,
        root=os.path.dirname(image_path) if not uses_absolute_paths else None,
        transforms=transform_list,
        loader=resilient_loader,
    )

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=not shuffle_off)
        sampler.set_epoch(get_rank())

    if shuffle_off:
        shuffle = False
    else:
        shuffle = sampler is None

    loader = DataLoader(
        dataset,
        config['batch_size'],
        shuffle=shuffle,
        drop_last=True,
        sampler=sampler,
    )
    return loader


def build_latent_and_noise_generator(autoencoder: StyleganAutoencoder, config: Dict) -> Iterable:
    torch.random.manual_seed(1)
    while True:
        latent_code = torch.randn(config['batch_size'], config['latent_size'])
        noise = autoencoder.decoder.make_noise()
        yield Latents(latent_code, noise)
