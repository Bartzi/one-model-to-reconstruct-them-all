import argparse
import imgaug.augmenters as iaa
from pathlib import Path

import numpy
from PIL import Image
from tqdm import tqdm


NOISE_SCALES = [5, 10, 15, 25, 35, 50]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="based on a given image dir, create noisy versions of images and save them in extra dirs + gt")
    parser.add_argument("image_dir")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)

    for scale in tqdm(NOISE_SCALES):
        image_files = list(image_dir.glob('*.png'))
        with Image.open(image_files[0]) as test_image:
            per_channel = test_image.mode != 'L'

        augmenter = iaa.AdditiveGaussianNoise(scale=scale, per_channel=per_channel)
        dest_dir = image_dir.parent / f"noisy_{scale}"
        dest_dir.mkdir(exist_ok=True)

        for image_file in tqdm(image_files, leave=False):
            with Image.open(image_file) as the_image:
                image_array = numpy.array(the_image)
            noisy_array = augmenter(image=image_array)
            noisy_image = Image.fromarray(noisy_array)

            dest_name = dest_dir / image_file.name
            noisy_image.save(str(dest_name))
