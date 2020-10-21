import math

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


class Encoder(nn.Module):

    def __init__(self, image_size: int, latent_size: int, num_input_channels: int, size_channel_map: dict, target_size: int = 4):
        super().__init__()

        self.image_size = image_size
        self.latent_size = latent_size
        log_input_size = int(math.log(image_size, 2))
        log_target_size = int(math.log(target_size, 2))
        assert image_size > target_size, "Input size must be larger than target size"
        assert 2 ** log_input_size == image_size, "Input size must be a power of 2"
        assert 2 ** log_target_size == target_size, "Target size must be a power of 2"

        self.start_block = BasicBlock(
            num_input_channels,
            size_channel_map[image_size],
            downsample=nn.Sequential(
                nn.Conv2d(num_input_channels, size_channel_map[image_size], kernel_size=1, stride=1),
                nn.BatchNorm2d(size_channel_map[image_size])
            )
        )

        self.resnet_blocks = [
            BasicBlock(
                in_planes := size_channel_map[2 ** current_size],
                out_planes := size_channel_map[2 ** (current_size - 1)],
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_planes)
                )
            )
            for current_size in range(log_input_size, log_target_size, -1)
        ]
        self.resnet_blocks = nn.ModuleList([self.start_block] + self.resnet_blocks)

        num_latents = (log_input_size - log_target_size) * 2 + 2
        self.to_latent = [
            nn.Conv2d(size_channel_map[target_size], self.latent_size, kernel_size=(target_size, target_size), stride=1)
            for _ in range(num_latents)
        ]
        self.to_latent = nn.ModuleList(self.to_latent)

    def forward(self, x):
        h = x
        for resnet_block in self.resnet_blocks:
            h = resnet_block(h)

        latent_codes = [to_latent(h) for to_latent in self.to_latent]
        latent_codes = torch.stack(latent_codes, dim=1)
        latent_codes = latent_codes.squeeze(3).squeeze(3)

        return latent_codes
