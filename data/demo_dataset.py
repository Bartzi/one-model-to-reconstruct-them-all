import numpy

from pytorch_training.data.utils import default_loader
from torch.utils import data
from typing import Callable, Dict


class DemoDataset(data.Dataset):

    def __init__(self, image_file: str, root: str = None, transforms: Callable = None, loader: Callable = default_loader):
        self.image_file = image_file
        self.transforms = transforms
        self.loader = loader

    def __len__(self):
        return 1

    def __getitem__(self, index: int) -> Dict[str, numpy.ndarray]:
        image = self.loader(self.image_file)

        if self.transforms is not None:
            image = self.transforms(image)

        return {
            'input_image': image,
            'output_image': image
        }

