import argparse
from pathlib import Path

import torch


def convert_autoencoder_checkpoint(checkpoint):
    encoder_weights = {}
    decoder_weights = {}
    autoencoder_weights = {}

    if all(key in checkpoint for key in ['encoder', 'decoder']):
        # already converted, no need for further conversion
        return checkpoint

    for name, weight in checkpoint['autoencoder'].items():
        name = name.split('.')

        if name[0] == 'module':
            name = name[1:]

        for name_part in ['encoder', 'decoder']:
            name_part_in_name = [n == name_part for n in name]
            if any(name_part_in_name):
                new_name = '.'.join(name[name_part_in_name.index(True) + 1:])
                eval(f"{name_part}_weights")[new_name] = weight
                break

        autoencoder_weights['.'.join(name)] =weight

    checkpoint['autoencoder'] = autoencoder_weights
    checkpoint['encoder'] = encoder_weights
    checkpoint['decoder'] = decoder_weights

    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert autoencoder checkpoint without decoder and encoder keys")
    parser.add_argument('checkpoint', help='path to checkpoint to convert')
    parser.add_argument('--destination', help='if you want to save it in another file (relative to original checkpoint)')

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    checkpoint = convert_autoencoder_checkpoint(checkpoint)

    dest_file_name = Path(args.checkpoint)
    if args.destination is not None:
        dest_file_name = dest_file_name.parent / args.destination

    torch.save(
        checkpoint,
        str(dest_file_name)
    )
