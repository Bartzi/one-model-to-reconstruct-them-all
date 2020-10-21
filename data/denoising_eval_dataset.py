import json
from typing import Callable, Dict

import numpy
import os
from torch.utils.data import Dataset

from pytorch_training.data.utils import default_loader


class DenoisingEvaluationDataset(Dataset):

    def __init__(self, json_file: str, root: str = None, transforms: Callable = None, loader: Callable = default_loader):
        with open(json_file) as f:
            self.image_data = json.load(f)

        self.root = root
        self.transforms = transforms
        self.loader = loader

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index: int) -> Dict[str, numpy.ndarray]:
        paths = self.image_data[index]

        loaded_images = {}
        for image_type, path in paths.items():
            if self.root is not None:
                path = os.path.join(self.root, path)

            image = self.loader(path)
            if self.transforms is not None:
                image = self.transforms(image)

            loaded_images[image_type] = image

        return loaded_images

