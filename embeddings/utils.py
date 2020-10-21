from typing import Dict, List

import numpy
import torch


def noises_from_embedding(embedding_data: Dict[str, numpy.ndarray], index: int) -> List[torch.Tensor]:
    noise_keys = [key for key in embedding_data.keys() if 'noise' in key]
    noise_keys = sorted(noise_keys, key=lambda x: (int((s := x.split('_'))[-2]), int(s[-1])))

    noises = [embedding_data[noise_key][index].astype(numpy.float32) for noise_key in noise_keys]
    noises = [torch.tensor(noise) for noise in noises]
    spreaded_noises = []
    for noise in noises:
        if noise.shape[0] > 1:
            # we are dealing with multiple noise maps per resolution, as it happens in Stylegan2
            spreaded_noises.extend(noise.chunk(2, dim=0))
        else:
            spreaded_noises.append(noise)
    return spreaded_noises


def latent_from_embedding(embedding_data: Dict[str, numpy.ndarray], index:int) -> torch.Tensor:
    return torch.tensor(embedding_data['latent_codes'][index].astype(numpy.float32))
