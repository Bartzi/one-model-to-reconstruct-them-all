import argparse
from pathlib import Path

from PIL import Image
from pytorch_training.images import make_image

from data.demo_dataset import DemoDataset
from networks import get_autoencoder, load_weights
from utils.config import load_config
from utils.data_loading import build_data_loader


def main(args):
    root_dir = Path(args.autoencoder_checkpoint).parent.parent
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    config = load_config(args.autoencoder_checkpoint, None)
    config['batch_size'] = 1
    autoencoder = get_autoencoder(config).to(args.device)
    autoencoder = load_weights(autoencoder, args.autoencoder_checkpoint, key='autoencoder')

    input_image = Path(args.image)
    data_loader = build_data_loader(input_image, config, config['absolute'], shuffle_off=True, dataset_class=DemoDataset)

    image = next(iter(data_loader))
    image = {k: v.to(args.device) for k,v in image.items()}

    reconstructed = Image.fromarray(make_image(autoencoder(image['input_image'])[0].squeeze(0)))

    output_name = Path(args.output_dir) / f"reconstructed_{input_image.stem}_stylegan_{config['stylegan_variant']}_{'w_only' if config['w_only'] else 'w_plus'}.png"
    reconstructed.save(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="reconstruct a given image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("autoencoder_checkpoint", help='Path to autoencoder checkpoint which shall be used for embedding')
    parser.add_argument("image", help="image to reconstruct")
    parser.add_argument("--device", default='cuda', help="which device to use (cuda, or cpu)")
    parser.add_argument("--output-dir", default='.')

    main(parser.parse_args())
