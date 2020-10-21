import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy
import torch
from PIL import Image
from tqdm import tqdm, trange

sys.path.append(str(Path(__file__).resolve().parent.parent))


from latent_projecting import Latents
from networks import load_weights, StyleganAutoencoder, get_autoencoder
from pytorch_training.images import make_image
from utils.config import load_config
from utils.data_loading import build_data_loader, build_latent_and_noise_generator


def noise_normalize(tensor: torch.Tensor) -> torch.Tensor:
    normalized_tensors = []
    for sub_tensor in tensor:
        min_value = sub_tensor.min()
        sub_tensor = sub_tensor.sub(min_value)
        max_value = sub_tensor.max()
        normalized_tensors.append(sub_tensor.div(max_value + 1e-8))

    return torch.stack(normalized_tensors)


def render_with_shifted_noise(autoencoder: StyleganAutoencoder, latents: Latents, shifting_rounds: int) -> List[List[Image.Image]]:
    if shifting_rounds == 1:
        shift_factor = torch.tensor([random.random() * 4 - 2])
    else:
        shift_factor = torch.tensor(numpy.linspace(-2, 2, num=shifting_rounds))

    def generate(latents: Latents) -> torch.Tensor:
        with torch.no_grad():
            generated, _ = autoencoder.decoder([latents.latent], input_is_latent=autoencoder.is_wplus(latents), noise=latents.noise)
        return generated

    shifted_images = [[Image.fromarray(make_image(generate(latents)[0]))] for _ in range(shifting_rounds)]

    for the_round in trange(shifting_rounds, leave=False):
        for i in range(len(latents.noise)):
            noise_copy = latents.noise[i].clone()
            latents.noise[i] = latents.noise[i] * shift_factor[the_round]
            generated_image = generate(latents)
            generated_image = Image.fromarray(make_image(generated_image[0]))
            shifted_images[the_round].append(generated_image)
            latents.noise[i] = noise_copy

    return shifted_images


def main(args):
    checkpoint_path = Path(args.model_checkpoint)

    config = load_config(checkpoint_path, None)

    autoencoder = get_autoencoder(config).to(args.device)
    load_weights(autoencoder, checkpoint_path, key='autoencoder', strict=True)

    config['batch_size'] = 1
    if args.generate:
        data_loader = build_latent_and_noise_generator(autoencoder, config)
    else:
        data_loader = build_data_loader(args.images, config, args.absolute, shuffle_off=True)

    noise_dest_dir = checkpoint_path.parent.parent / "noise_maps"
    noise_dest_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    for idx, batch in enumerate(tqdm(data_loader, total=args.num_images)):
        batch = batch.to(args.device)

        if args.generate:
            latents = batch
            image_names = [Path(f"generate_{idx}.png")]
        else:
            with torch.no_grad():
                latents: Latents = autoencoder.encode(batch)

            image_names = [Path(data_loader.dataset.image_data[idx * config['batch_size'] + batch_idx]) for batch_idx in range(len(batch))]

        if args.shift_noise:
            noise_shifted_tensors = render_with_shifted_noise(autoencoder, latents, args.rounds)

        images = []
        for noise_tensors in latents.noise:
            noise_images = make_image(noise_tensors, normalize_func=noise_normalize)
            images.append([Image.fromarray(im).resize((config['image_size'], config['image_size']), Image.NEAREST) for im in noise_images])

        for batch_idx, (image, orig_file_name) in enumerate(zip(batch, image_names)):
            full_image = Image.new(
                'RGB',
                (
                    (len(images) + 1) * config['image_size'],
                    config['image_size'] if not args.shift_noise else config['image_size'] * (args.rounds + 1)
                )
            )
            if not args.generate:
                full_image.paste(Image.fromarray(make_image(image)), (0, 0))
            for i, noise_images in enumerate(images):
                full_image.paste(noise_images[batch_idx], ((i + 1) * config['image_size'], 0))

            if args.shift_noise:
                for i, shifted_images in enumerate(noise_shifted_tensors):
                    for j, shifted_image in enumerate(shifted_images):
                        full_image.paste(shifted_image, (j * config['image_size'], (i + 1) * config['image_size']))

            full_image.save(noise_dest_dir / f"{orig_file_name.stem}_noise.png")

        num_images += len(image_names)
        if num_images >= args.num_images:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render noise maps of given image")
    parser.add_argument("model_checkpoint", help="path to model checkpoint that is to be used to generate noise map")
    parser.add_argument("images", help="path to json with all images to be analyzed")
    parser.add_argument("-n", "--num-images", type=int, default=1, help="number of images where you want to have a look at the noise maps")
    parser.add_argument("--absolute", action='store_true', default=False, help="use this if the json contains absolute paths")
    parser.add_argument("--device", default='cuda', help="which device to use")
    parser.add_argument("-s", "--shift-noise", action='store_true', default=False, help="do not just render the noise maps, but also render some versions of the image with shifted noise vectors")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="Number of shifting rounds to perform")
    parser.add_argument("--generate", action='store_true', default=False, help="Do not use images, but use unconditional generation instead")

    main(parser.parse_args())
