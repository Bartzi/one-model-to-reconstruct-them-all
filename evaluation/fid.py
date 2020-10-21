import contextlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple

import numpy
import pytorch_fid.fid_score
import pytorch_fid.inception
import torch
import torch.distributed
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from tqdm.contrib import tenumerate

from networks import StyleganAutoencoder
from pytorch_training.data.utils import cache_or_load_file
from pytorch_training.distributed import get_world_size, get_rank


@dataclass
class FIDStatistics:
    mu: numpy.ndarray
    sigma: numpy.ndarray


class FID:

    def __init__(self, num_samples: int = 1000, dim: int = 2048, device: str = 'cuda'):
        self.num_samples = num_samples
        self.inception_dim = dim
        self.block_index = pytorch_fid.inception.InceptionV3.BLOCK_INDEX_BY_DIM[dim]
        self.device = device

        self.model = pytorch_fid.inception.InceptionV3([self.block_index])
        self.model.eval()

    @contextlib.contextmanager
    def init_inception_model(self):
        torch.cuda.empty_cache()
        self.model = self.model.to(self.device)
        yield
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()

    def __call__(self, model: StyleganAutoencoder, data_loader: DataLoader, dataset_path: Union[str, Path] = None) -> float:
        with self.init_inception_model():
            real_statistics, fake_statistics = self.calculate_statistics(model, data_loader, dataset_path)

        fid_score = pytorch_fid.fid_score.calculate_frechet_distance(
            real_statistics.mu,
            real_statistics.sigma,
            fake_statistics.mu,
            fake_statistics.sigma
        )

        return fid_score

    @staticmethod
    def load_precalculated_mu_and_sigma(path: Path) -> FIDStatistics:
        data = numpy.load(str(path))
        return FIDStatistics(data['mu'][:], data['sigma'][:])

    @staticmethod
    def get_statistics(activations: numpy.ndarray) -> FIDStatistics:
        mu = numpy.mean(activations, axis=0)
        sigma = numpy.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def multiprocess_synchronize(self, activations: torch.Tensor) -> numpy.ndarray:
        if get_world_size() > 1:
            # we are running in distributed setting, so we will need to gather all predictions for each worker
            gathered_activations = [torch.empty(activations.shape, device=self.device) for _ in range(get_world_size())]
            torch.distributed.all_gather(gathered_activations, activations)
            activations = torch.cat(gathered_activations, dim=0)
        return activations.cpu().numpy()

    @contextlib.contextmanager
    def get_progress_bar(self, data_loader, total, description):
        if get_rank() == 0:
            pbar = tenumerate(data_loader, total=total // data_loader.batch_size + 1, desc=description, leave=False)
        else:
            pbar = enumerate(data_loader)

        yield pbar

        if hasattr(pbar, 'close'):
            pbar.close()

    def calculate_real_statistics(self, path: Union[Path, None], data_loader: DataLoader) -> FIDStatistics:
        total = min(self.num_samples, len(data_loader) * data_loader.batch_size)
        activations = torch.empty((total, self.inception_dim), device=self.device)

        with self.get_progress_bar(data_loader, total, 'fid real') as progress_bar:
            for i, batch in progress_bar:
                batch = batch['output_image']
                batch = batch.to(self.device)
                inception_predictions = self.get_inception_predictions(batch)

                start = i * data_loader.batch_size
                end = min((i + 1) * data_loader.batch_size, total)
                activations[start:end] = inception_predictions[:end-start]

                if end >= total:
                    break

        activations = self.multiprocess_synchronize(activations)
        statistics = self.get_statistics(activations)
        if path is not None:
            numpy.savez(str(path), mu=statistics.mu, sigma=statistics.sigma)

        return statistics

    def get_inception_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inception_predictions = self.model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if inception_predictions.size(2) != 1 or inception_predictions.size(3) != 1:
            inception_predictions = adaptive_avg_pool2d(inception_predictions, output_size=(1, 1))
        inception_predictions = inception_predictions
        inception_predictions = inception_predictions.reshape(len(inception_predictions), -1)

        return inception_predictions

    def calculate_fake_statistics(self, autoencoder: StyleganAutoencoder, data_loader: DataLoader) -> FIDStatistics:
        total = min(self.num_samples, len(data_loader) * data_loader.batch_size)
        activations = torch.empty((total, self.inception_dim), device=self.device)

        with self.get_progress_bar(data_loader, total, 'fid fake') as progress_bar:
            for i, batch in progress_bar:
                batch = batch['input_image']
                batch = batch.to(self.device)
                with torch.no_grad():
                    reconstructed_images = autoencoder(batch)

                inception_predictions = self.get_inception_predictions(reconstructed_images)

                start = i * data_loader.batch_size
                end = min((i + 1) * data_loader.batch_size, total)
                activations[start:end] = inception_predictions[:end-start]

                if end >= total:
                    break

        activations = self.multiprocess_synchronize(activations)
        return self.get_statistics(activations)

    def calculate_statistics(self, autoencoder: StyleganAutoencoder, data_loader: DataLoader, dataset_path: Union[str, Path]) -> Tuple[FIDStatistics, FIDStatistics]:
        if dataset_path is not None:
            dataset_path = Path(dataset_path)

            hasher = hashlib.sha512(str(dataset_path).encode('utf-8'))
            hasher.update(str(self.num_samples).encode('utf-8'))
            fid_file_name = f"{hasher.hexdigest()}_fid.npz"
            real_statistics = cache_or_load_file(
                dataset_path.parent / fid_file_name,
                lambda x: self.calculate_real_statistics(x, data_loader),
                self.load_precalculated_mu_and_sigma
            )
        else:
            real_statistics = self.calculate_real_statistics(None, data_loader)

        torch.cuda.empty_cache()

        fake_statistics = self.calculate_fake_statistics(autoencoder, data_loader)

        torch.cuda.empty_cache()

        return real_statistics, fake_statistics
