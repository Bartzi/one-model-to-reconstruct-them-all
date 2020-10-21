import argparse
from argparse import Namespace
from pathlib import Path

from PIL import Image

from latent_projecting.style_transfer import StyleTransferer
from utils.command_line_args import add_default_args_for_projecting
from pytorch_training.images.utils import make_image


def main(args: Namespace):
    transferer = StyleTransferer(args)
    content_latents, style_latents = transferer.get_latents(args.content_path, args.style_path)

    if args.mixing_index < 0:
        stylized_images = {
            i: transferer.do_style_transfer(content_latents, style_latents, i)
            for i in range(content_latents.latent.shape[1])
        }
    else:
        stylized_images = {
            args.mixing_index: transferer.do_style_transfer(content_latents, style_latents, args.mixing_index)
        }

    destination_dir = Path(args.content_path).parent / "simple_style_transfer" / args.destination_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    for index, (image_array, optimization_path) in stylized_images.items():
        content_base_name = args.content_path
        style_base_name = args.style_path

        content_name = Path(content_base_name).stem
        style_name = Path(style_base_name).stem

        image_name = f"{content_name}_{style_name}_{index}"
        destination_name = destination_dir / f"{image_name}.png"
        Image.fromarray(make_image(image_array)[0]).save(destination_name)

        if optimization_path is not None and args.gif:
            transferer.projector.create_gif(optimization_path.latent, optimization_path.noise, image_name, destination_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that does style transfer described in Image2Stylegan Paper")
    parser.add_argument("--content", dest="content_path", required=True)
    parser.add_argument("--style", dest="style_path", required=True)
    parser.add_argument("--destination", dest="destination_dir", required=True)
    parser.add_argument("--mixing-index", type=int, default=-1)
    parser.add_argument("--post-optimize", action='store_true', default=False)
    parser.add_argument("--gif", action='store_true', default=False)
    parser = add_default_args_for_projecting(parser)

    main(parser.parse_args())
