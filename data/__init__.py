import argparse
from typing import Type

from data.autoencoder_dataset import AutoencoderDataset, DenoisingAutoencoderDataset, \
    BlackAndWhiteDenoisingAutoencoderDataset


def get_dataset_class(args: argparse.Namespace) -> Type[AutoencoderDataset]:
    if getattr(args, 'denoising', False):
        dataset_class = DenoisingAutoencoderDataset
    elif getattr(args, 'black_and_white_denoising', False):
        dataset_class = BlackAndWhiteDenoisingAutoencoderDataset
    else:
        dataset_class = AutoencoderDataset
    return dataset_class
