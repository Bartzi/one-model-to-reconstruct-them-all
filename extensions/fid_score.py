from pathlib import Path

from pytorch_training.distributed import get_rank, synchronize
from pytorch_training.reporter import get_current_reporter
from torch.utils.data import DataLoader
from typing import Union

from evaluation.fid import FID
from networks import StyleganAutoencoder

from pytorch_training import Extension, Trainer


class FIDScore(Extension):

    def __init__(self, autoencoder: StyleganAutoencoder, data_loader: DataLoader, *args, dataset_path: Union[str, Path] = None, device: str = 'cuda', **kwargs):
        super().__init__(*args, **kwargs)
        self.fid_calculator = FID(device=device)
        self.autoencoder = autoencoder
        self.data_loader = data_loader
        self.dataset_path = dataset_path

    def initialize(self, trainer: 'Trainer'):
        self.run(trainer)

    def finalize(self, trainer: 'Trainer'):
        self.run(trainer)

    def run(self, trainer: Trainer):
        fid = self.fid_calculator(self.autoencoder, self.data_loader, self.dataset_path)
        synchronize()
        if get_rank() == 0:
            with get_current_reporter() as reporter:
                reporter.add_observation({"fid_score": fid}, "evaluation")
