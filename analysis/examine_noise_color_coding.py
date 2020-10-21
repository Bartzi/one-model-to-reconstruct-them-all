import argparse
import sys
from pathlib import Path

import numpy
import torch
from PIL import Image
from tqdm import tqdm
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from latent_projecting import Latents
from networks import get_stylegan_1_based_autoencoder, get_stylegan_2_based_autoencoder, load_weights, \
    StyleganAutoencoder
from pytorch_training.images import make_image
from utils.config import load_config
from utils.data_loading import build_data_loader, build_latent_and_noise_generator


def render_color_grid(autoencoder: StyleganAutoencoder, latents: Latents, indices: List[int], grid_size: int, bounds: List[int]) -> List[List[torch.Tensor]]:

    def generate(latents: Latents) -> torch.Tensor:
            with torch.no_grad():
                generated, _ = autoencoder.decoder([latents.latent], input_is_latent=autoencoder.is_wplus(latents), noise=latents.noise)
            return generated

    assert len(indices) == 2, "Render Color grid only supports the rendering of two indices at once!"
    assert len(bounds) == 2, "Render Color grid only supports the rendering with min and max bound"

    shift_factor = numpy.linspace(bounds[0], bounds[1], num=grid_size)
    x_shifts, y_shifts = map(numpy.squeeze, numpy.meshgrid(shift_factor, shift_factor, sparse=True))

    x_noise_map = latents.noise[indices[0]].clone()
    y_noise_map = latents.noise[indices[1]].clone()

    grid = []
    for y_shift in tqdm(y_shifts, leave=False):
        latents.noise[indices[1]] = y_noise_map.clone() * y_shift
        x_images = []
        for x_shift in tqdm(x_shifts, leave=False):
            latents.noise[indices[0]] = x_noise_map.clone() * x_shift
            generated_image = generate(latents)
            generated_image = Image.fromarray(make_image(generated_image[0]))
            x_images.append(generated_image)
        grid.append(x_images)

    return grid


def main(args):
    checkpoint_path = Path(args.model_checkpoint)

    config = load_config(checkpoint_path, None)
    if config['stylegan_variant'] == 1:
        autoencoder_func = get_stylegan_1_based_autoencoder(argparse.Namespace(**config))
    else:
        autoencoder_func = get_stylegan_2_based_autoencoder(argparse.Namespace(**config))

    autoencoder = autoencoder_func(
        config['image_size'],
        config['latent_size'],
        config['input_dim'],
    ).to(args.device)

    load_weights(autoencoder, checkpoint_path, key='autoencoder', strict=True)

    config['batch_size'] = 1
    if args.generate:
        data_loader = build_latent_and_noise_generator(autoencoder, config)
    else:
        data_loader = build_data_loader(args.images, config, args.absolute, shuffle_off=True)

    noise_dest_dir = checkpoint_path.parent.parent / "color_model_analysis"
    noise_dest_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    for idx, batch in enumerate(tqdm(data_loader, total=args.num_images)):
        batch = batch.to(args.device)
        if args.generate:
            latents = batch
            image_name = Path(f"generate_{idx}.png")
        else:
            with torch.no_grad():
                latents: Latents = autoencoder.encode(batch)

            image_name = Path(data_loader.dataset.image_data[idx])

        color_grid = render_color_grid(autoencoder, latents, args.indices, args.grid_size, args.bounds)

        full_image = Image.new(
            'RGB',
            (args.grid_size * config['image_size'], args.grid_size * config['image_size'])
        )

        for y, x_images in enumerate(color_grid):
            for x, image in enumerate(x_images):
                full_image.paste(image, (x * config['image_size'], y * config['image_size']))

            full_image.save(noise_dest_dir / f"{image_name.stem}_color_grid.png")

        num_images += 1
        if num_images >= args.num_images:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render noise maps of given image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_checkpoint", help="path to model checkpoint that is to be used to generate noise map")
    parser.add_argument("images", help="path to json with all images to be analyzed")
    parser.add_argument("-n", "--num-images", type=int, default=1, help="number of images where you want to have a look at the noise maps")
    parser.add_argument("--absolute", action='store_true', default=False, help="use this if the json contains absolute paths")
    parser.add_argument("--device", default='cuda', help="which device to use")
    parser.add_argument("-i", "--indices", type=int, nargs=2, default=[4, 5], help="indices to use for color space analysis")
    parser.add_argument("-g", "--grid-size", type=int, default=10, help="Size of rendered image grid (squared, so only one dim necessary)")
    parser.add_argument("-b", "--bounds", type=int, nargs=2, default=[-2, 2], help="interpolation bounds")
    parser.add_argument("--generate", action='store_true', default=False, help="Do not use images, but use unconditional generation instead")

    main(parser.parse_args())
