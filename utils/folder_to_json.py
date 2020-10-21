import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from pytorch_training.images import is_image


def get_file_name(root: str) -> str:
    for dir, _, files in os.walk(root):
        for file_name in files:
            if is_image(file_name) and not "embed-test" in dir:
                yield os.path.relpath(os.path.join(dir, file_name), start=root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert files in dir to json for training")
    parser.add_argument("dir")
    parser.add_argument("--split", action='store_true', default=False, help='create train and val split')

    args = parser.parse_args()

    files = [name for name in tqdm(get_file_name(args.dir))]

    dest_dir = Path(args.dir)
    if args.split:
        split_index = int(len(files) * 0.9)
        train_data = files[:split_index]
        val_data = files[split_index:]

        with (dest_dir / 'train.json').open('w') as f:
            json.dump(train_data, f, indent='\t')

        with (dest_dir / 'val.json').open('w') as f:
            json.dump(val_data, f, indent='\t')
    else:
        with open(os.path.join(args.dir, "images.json"), 'w') as f:
            json.dump(files, f, indent='\t')
