import functools
import os
from typing import Dict

import imgaug
import imgaug.augmenters as iaa
import numpy
from PIL import Image

from pytorch_training.data.json_dataset import JSONDataset

DENOISING_VARIANCES = [5, 10, 15, 25, 35, 50]
imgaug.seed(666)


class AutoencoderDataset(JSONDataset):

    def augment_image(self, image: Image) -> Image:
        return image

    def __getitem__(self, index: int) -> Dict[str, numpy.ndarray]:
        path = self.image_data[index]
        if self.root is not None:
            path = os.path.join(self.root, path)

        image = self.loader(path)
        augmented_image = self.augment_image(image)

        if self.transforms is not None:
            image = self.transforms(image)
            augmented_image = self.transforms(augmented_image)

        return {
            'input_image': augmented_image,
            'output_image': image
        }


class DenoisingAutoencoderDataset(AutoencoderDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gaussian_noise = functools.partial(iaa.AdditiveGaussianNoise, scale=DENOISING_VARIANCES)
        self.noise_augmenter = iaa.OneOf([
            gaussian_noise(),
            gaussian_noise(per_channel=True)
        ])

    def augment_image(self, image: Image) -> Image:
        image = numpy.array(image).copy()
        image = self.noise_augmenter(image=image)
        image = Image.fromarray(image)
        return image


class BlackAndWhiteDenoisingAutoencoderDataset(DenoisingAutoencoderDataset):

    def __init__(self, *args, **kwargs):
        loader_func = kwargs['loader']
        kwargs['loader'] = lambda path: loader_func(path).convert('L').convert('RGB')
        super().__init__(*args, **kwargs)

    def augment_image(self, image: Image) -> Image:
        image = super().augment_image(image)
        image = image.convert('L').convert("RGB")
        return image
