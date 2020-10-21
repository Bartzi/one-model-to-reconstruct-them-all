import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock

from latent_projecting import Latents, CodeLatents


class UNetLikeEncoder(nn.Module):

    def __init__(self, image_size: int, latent_size: int, num_input_channels: int, size_channel_map: dict, *, target_size: int = 4, stylegan_variant: int = 2):
        super().__init__()

        self.image_size = image_size
        self.latent_size = latent_size
        self.stylegan_variant = stylegan_variant
        self.size_channel_map = size_channel_map

        self.log_input_size = int(math.log(image_size, 2))
        self.log_target_size = int(math.log(target_size, 2))
        assert image_size > target_size, "Input size must be larger than target size"
        assert 2 ** self.log_input_size == image_size, "Input size must be a power of 2"
        assert 2 ** self.log_target_size == target_size, "Target size must be a power of 2"

        self.start_block = BasicBlock(
            num_input_channels,
            size_channel_map[image_size],
            downsample=nn.Sequential(
                nn.Conv2d(num_input_channels, size_channel_map[image_size], kernel_size=1, stride=1),
                nn.BatchNorm2d(size_channel_map[image_size])
            )
        )
        self.intermediate_block = BasicBlock(
            size_channel_map[image_size],
            size_channel_map[image_size],
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
            for current_size in range(self.log_input_size, self.log_target_size, -1)
        ]

        self.intermediate_resnet_blocks = [
            BasicBlock(
                in_planes := size_channel_map[2 ** current_size],
                in_planes,
            )
            for current_size in range(self.log_input_size, self.log_target_size - 1, -1)
        ]

        self.resnet_blocks = nn.ModuleList([self.start_block] + self.resnet_blocks)
        self.intermediate_resnet_blocks = nn.ModuleList(self.intermediate_resnet_blocks)

        num_latents = (self.log_input_size - self.log_target_size) * 2 + 2
        assert sum(map(len, [self.resnet_blocks, self.intermediate_resnet_blocks])) == num_latents, "The sum of all resnet blocks must be equal to the number of required latents"

        self.build_projecting_layers(self.log_input_size, self.log_target_size, size_channel_map)

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        raise NotImplementedError

    def get_to_x_convs(self, input_size: int, target_size: int, target_channels: int, size_channel_map: Dict[int, int]) -> nn.ModuleList:
        conv_layer = [
            nn.Conv2d(size_channel_map[2 ** current_size], target_channels, kernel_size=1, stride=1)
            for current_size in range(input_size, target_size - 1, -1)
        ]
        return nn.ModuleList(conv_layer)

    def forward(self, x: torch.Tensor) -> Latents:
        raise NotImplementedError


class WPlusEncoder(UNetLikeEncoder):

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        self.to_latent = self.get_to_x_convs(log_input_size, log_target_size, self.latent_size, size_channel_map)
        self.intermediate_to_latent = self.get_to_x_convs(log_input_size, log_target_size, self.latent_size, size_channel_map)

        self.to_noise = self.get_to_x_convs(log_input_size, log_target_size, 1, size_channel_map)
        if self.stylegan_variant == 2:
            self.intermediate_to_noise = self.get_to_x_convs(log_input_size, log_target_size, 1, size_channel_map)

    def forward(self, x: torch.Tensor) -> Latents:
        latent_codes = []
        noise_codes = []
        h = x

        for i in range(len(self.resnet_blocks)):
            h = self.resnet_blocks[i](h)
            latent_codes.append(self.to_latent[i](F.adaptive_avg_pool2d(h, (1, 1))))
            noise_codes.append(self.to_noise[i](h))
            h = self.intermediate_resnet_blocks[i](h)
            latent_codes.append(self.intermediate_to_latent[i](F.adaptive_avg_pool2d(h, (1, 1))))
            if self.stylegan_variant == 2 and i < len(self.resnet_blocks) - 1:
                noise_codes.append(self.intermediate_to_noise[i](h))

        latent_codes.reverse()
        latent_codes = torch.stack(latent_codes, dim=1)
        latent_codes = latent_codes.squeeze(3).squeeze(3)

        noise_codes.reverse()

        return Latents(latent_codes, noise_codes)


class WPlusResnetNoiseEncoder(WPlusEncoder):

    def get_noise_resblocks(self, input_size: int, target_size: int, size_channel_map: Dict[int, int]) -> nn.ModuleList:
        resblocks = [
            BasicBlock(
                in_planes := size_channel_map[2 ** current_size],
                1,
                downsample=nn.Sequential(
                    nn.Conv2d(in_planes, 1, kernel_size=1, stride=1)
                )
            )
            for current_size in range(input_size, target_size - 1, -1)
        ]
        return nn.ModuleList(resblocks)

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        self.to_latent = self.get_to_x_convs(log_input_size, log_target_size, self.latent_size, size_channel_map)
        self.intermediate_to_latent = self.get_to_x_convs(log_input_size, log_target_size, self.latent_size, size_channel_map)

        self.to_noise = self.get_noise_resblocks(log_input_size, log_target_size, size_channel_map)
        if self.stylegan_variant == 2:
            self.intermediate_to_noise = self.get_noise_resblocks(log_input_size, log_target_size, size_channel_map)


class WEncoder(UNetLikeEncoder):

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        self.to_latent = nn.Conv2d(self.latent_size, self.latent_size, kernel_size=1, stride=1)
        self.to_noise = self.get_to_x_convs(log_input_size, log_target_size, 1, size_channel_map)
        if self.stylegan_variant == 2:
            self.intermediate_to_noise = self.get_to_x_convs(log_input_size, log_target_size, 1, size_channel_map)

    def forward(self, x: torch.Tensor) -> Latents:
        latent_codes = []
        noise_codes = []
        h = x

        for i in range(len(self.resnet_blocks)):
            h = self.resnet_blocks[i](h)
            noise_codes.append(self.to_noise[i](h))
            h = self.intermediate_resnet_blocks[i](h)
            if self.stylegan_variant == 2 and i < len(self.resnet_blocks) - 1:
                noise_codes.append(self.intermediate_to_noise[i](h))

        latent_codes.append(self.to_latent(F.adaptive_avg_pool2d(h, (1, 1))))

        latent_codes.reverse()
        latent_codes = latent_codes[0].squeeze(2).squeeze(2)

        noise_codes.reverse()

        return Latents(latent_codes, noise_codes)


class WWPlusEncoder(WPlusEncoder):

    def forward(self, x: torch.Tensor) -> Latents:
        resulting_latents = super().forward(x)
        latent_code = resulting_latents.latent.sum(dim=1)
        return Latents(latent_code, resulting_latents.noise)


class WCodeEncoder(WEncoder):

    def __init__(self, code_dim: int, /, *args, **kwargs):
        self.code_dim = code_dim
        super().__init__(*args, **kwargs)

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        super().build_projecting_layers(log_input_size, log_target_size, size_channel_map)
        self.to_code = nn.Conv2d(self.latent_size, self.code_dim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> CodeLatents:
        noise_codes = []
        h = x

        for i in range(len(self.resnet_blocks)):
            h = self.resnet_blocks[i](h)
            noise_codes.append(self.to_noise[i](h))
            h = self.intermediate_resnet_blocks[i](h)
            if self.stylegan_variant == 2 and i < len(self.resnet_blocks) - 1:
                noise_codes.append(self.intermediate_to_noise[i](h))

        h = F.adaptive_avg_pool2d(h, (1, 1))

        latent_code = self.to_latent(h)
        latent_code = latent_code.squeeze(2).squeeze(2)

        info_code = self.to_code(h)
        info_code = info_code.squeeze(2).squeeze(2)

        noise_codes.reverse()

        return CodeLatents(latent_code, noise_codes, info_code)


class WPlusNoNoiseEncoder(UNetLikeEncoder):

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        self.to_latent = self.get_to_x_convs(log_input_size, log_target_size, self.latent_size, size_channel_map)
        self.intermediate_to_latent = self.get_to_x_convs(log_input_size, log_target_size, self.latent_size, size_channel_map)

    def forward(self, x: torch.Tensor) -> Latents:
        latent_codes = []
        h = x

        for i in range(len(self.resnet_blocks)):
            h = self.resnet_blocks[i](h)
            latent_codes.append(self.to_latent[i](F.adaptive_avg_pool2d(h, (1, 1))))
            h = self.intermediate_resnet_blocks[i](h)
            latent_codes.append(self.intermediate_to_latent[i](F.adaptive_avg_pool2d(h, (1, 1))))

        latent_codes.reverse()
        latent_codes = torch.stack(latent_codes, dim=1)
        latent_codes = latent_codes.squeeze(3).squeeze(3)

        return Latents(latent_codes, None)


class WNoNoiseEncoder(WPlusNoNoiseEncoder):

    def forward(self, x: torch.Tensor) -> Latents:
        resulting_latents = super().forward(x)
        latent_code = resulting_latents.latent.sum(dim=1)
        return Latents(latent_code, resulting_latents.noise)


class NoiseEncoder(UNetLikeEncoder):

    def build_projecting_layers(self, log_input_size, log_target_size, size_channel_map):
        self.to_noise = self.get_to_x_convs(log_input_size, log_target_size, 1, size_channel_map)
        if self.stylegan_variant == 2:
            self.intermediate_to_noise = self.get_to_x_convs(log_input_size, log_target_size, 1, size_channel_map)

    def forward(self, x: torch.Tensor) -> Latents:
        noise_codes = []
        h = x

        for i in range(len(self.resnet_blocks)):
            h = self.resnet_blocks[i](h)
            noise_codes.append(self.to_noise[i](h))
            h = self.intermediate_resnet_blocks[i](h)
            if self.stylegan_variant == 2 and i < len(self.resnet_blocks) - 1:
                noise_codes.append(self.intermediate_to_noise[i](h))

        noise_codes.reverse()

        return Latents(None, noise_codes)
