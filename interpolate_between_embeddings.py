import argparse
import random
from pathlib import Path
from typing import Union, Tuple, Dict, List

import numpy
import torch
from PIL import Image
from torch import nn
from tqdm import trange

from embeddings.utils import latent_from_embedding, noises_from_embedding
from networks import get_autoencoder, load_weights
from pytorch_training.images import make_image
from utils.config import load_config


def interpolate(start_array: Union[numpy.ndarray, torch.Tensor], end_array: Union[numpy.ndarray, torch.Tensor], fraction: float) -> Union[numpy.ndarray, torch.Tensor]:
    return (1 - fraction) * start_array + fraction * end_array


def load_embeddings(embeddings: Dict[str, numpy.ndarray], index: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    latent = latent_from_embedding(embeddings, index).unsqueeze(0)
    noises = [noise.unsqueeze(0) for noise in noises_from_embedding(embeddings, index)]
    return latent, noises


def make_interpolation_image(steps: int, device: torch.device, autoencoder: nn.Module, is_w_plus: bool,
                            start_latent: torch.Tensor, end_latent: torch.Tensor,
                            start_noises: List[torch.Tensor], end_noises: List[torch.Tensor]):
    all_interpolation_images = []
    for interpolation_strategy in ['all', 'latent', 'noise']:
        interpolation_images = []

        start_image, _ = autoencoder.decoder([start_latent.to(device)], input_is_latent=is_w_plus, noise=[n.to(device) for n in start_noises])
        interpolation_images.append(make_image(start_image.squeeze(0)))

        for i in trange(steps + 1):
            step_fraction = i / steps
            if interpolation_strategy in ['latent', 'all']:
                latent = interpolate(start_latent, end_latent, step_fraction)
            else:
                latent = start_latent
            latent = latent.to(device)

            if interpolation_strategy in ['noise', 'all']:
                noises = [interpolate(start_noise, end_noise, step_fraction) for start_noise, end_noise in zip(start_noises, end_noises)]
            else:
                noises = autoencoder.decoder.make_noise()
            noises = [noise.to(device) for noise in noises]

            image, _ = autoencoder.decoder([latent], input_is_latent=is_w_plus, noise=noises)
            image = make_image(image.squeeze(0))
            interpolation_images.append(image)

        end_image, _ = autoencoder.decoder([end_latent.to(device)], input_is_latent=is_w_plus, noise=[n.to(device) for n in end_noises])
        interpolation_images.append(make_image(end_image.squeeze(0)))

        all_images = numpy.concatenate(interpolation_images, axis=1)
        image = Image.fromarray(all_images)
        all_interpolation_images.append(image)

    dest_image = Image.new("RGB", (all_interpolation_images[0].width, all_interpolation_images[0].height * 3))
    for i, image in enumerate(all_interpolation_images):
        dest_image.paste(image, (0, i * image.height))

    return dest_image


def main(args):
    embedding_dir = Path(args.embedding_file).parent
    embedded_data = numpy.load(args.embedding_file, mmap_mode='r')

    checkpoint_for_embedding = embedding_dir.parent / 'checkpoints' / f"{Path(args.embedding_file).stem.split('_')[-3]}.pt"

    config = load_config(checkpoint_for_embedding, None)
    autoencoder = get_autoencoder(config).to(args.device)
    autoencoder = load_weights(autoencoder, checkpoint_for_embedding, key='autoencoder', strict=True)

    num_images = len(embedded_data['image_names'])

    interpolation_dir = embedding_dir / 'interpolations'
    interpolation_dir.mkdir(parents=True, exist_ok=True)

    is_w_plus = not config['w_only']

    for _ in range(args.num_images):
        start_image_idx, end_image_idx = random.sample(list(range(num_images)), k=2)

        start_latent, start_noises = load_embeddings(embedded_data, start_image_idx)
        end_latent, end_noises = load_embeddings(embedded_data, end_image_idx)

        for steps in args.steps:
            result = make_interpolation_image(steps, args.device, autoencoder, is_w_plus,
                                              start_latent, end_latent, start_noises, end_noises)
            result.save(str(interpolation_dir / f"{start_image_idx}_{end_image_idx}_all_{steps}_steps.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract two embedding codes and interpolate between them based on a number of steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("embedding_file", help='Path to npz with embedding of latent codes + noise')
    parser.add_argument("--device", default='cuda', help="which device to use (cuda, or cpu)")
    parser.add_argument("-s", "--steps", type=int, default=[5, 20], nargs="+",
                        help="number of interpolation steps to perform (multiple values will create multiple scales)")
    parser.add_argument("-n", "--num-images", type=int, default=1,
                        help="perform interpolation or multiple images")

    main(parser.parse_args())
