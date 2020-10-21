import torch
from torch import Tensor


def gram_matrix(features: Tensor, mask: Tensor) -> Tensor:
    batch_size, num_channels, height, width = features.shape

    if mask is not None:
        normalize_denominator = mask.square().sum(dim=(2, 3)).sqrt()
        normalize_denominator = normalize_denominator.expand(1, 1, -1, -1)
        normalize_denominator = normalize_denominator.permute((2, 3, 0, 1))
        normalize_denominator = normalize_denominator.repeat((1,) + mask.shape[1:])
        normalized_mask = mask / normalize_denominator
        features = normalized_mask * features

    features = features.view(batch_size * num_channels, height * width)
    features = features.permute((1, 0))
    return torch.mm(features.T, features)


def euclidean_distance(tensor_1: Tensor, tensor_2: Tensor, mask: Tensor = None) -> Tensor:
    difference = tensor_1 - tensor_2
    if mask is not None:
        difference = mask * difference

    distance = difference.square().sum().sqrt() / tensor_1.shape.numel()
    return distance
