from typing import Tuple, Union

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from losses.perceptual_loss import PerceptualLoss
from losses.style_loss import StyleLoss


class AbstractPerceptualAndStyleLoss(nn.Module):

    def adapt_vgg_model(self, vgg: nn.Module) -> nn.Module:
        layers = []
        for layer in vgg.children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            layers.append(layer)
        return nn.Sequential(*layers)

    def disable_update(self):
        for parameter in self.parameters():
            parameter.requires_grad = False


class FixedPerceptualAndStyleLoss(AbstractPerceptualAndStyleLoss):

    def __init__(self, perceptual_target, style_target, perceptual_mask=None, style_mask=None):
        super().__init__()
        self.perceptual_target = perceptual_target
        self.style_target = style_target
        self.perceptual_mask = perceptual_mask
        self.style_mask = style_mask

        vgg = torchvision.models.vgg16(pretrained=True)
        vgg_features = self.adapt_vgg_model(vgg.features)

        blocks = {
            'conv1_1': vgg_features[:2],
            'conv1_2': vgg_features[2:4],
            'conv2_2': vgg_features[4:9],
            'conv3_3': vgg_features[9:16],
        }
        self.vgg_blocks = nn.ModuleDict(blocks)
        self.vgg_blocks.eval()

        self.style_losses = None
        self.perceptual_losses = None

        self.disable_update()

    def create_losses(self, target, mask, steps, loss_class) -> dict:
        features = target.clone()
        if mask is not None:
            mask = mask.clone()
        losses = {}

        for name, block in self.vgg_blocks.items():
            features = block(features)
            if mask is not None:
                mask = F.interpolate(mask, size=features.shape[-2:], mode='nearest')
            if name in steps:
                loss = loss_class(features, mask=mask)
                losses[name] = loss

        return losses

    def build_style_and_perceptual_losses(self):
        self.style_losses = nn.ModuleDict(
            self.create_losses(self.style_target, self.style_mask, ['conv3_3'], StyleLoss)
        )
        self.perceptual_losses = nn.ModuleDict(
            self.create_losses(self.perceptual_target, self.perceptual_mask, list(self.vgg_blocks.keys()), PerceptualLoss)
        )
        self.disable_update()

    def forward(self, generated_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.style_losses is None:
            self.build_style_and_perceptual_losses()

        features = generated_image
        style_losses = []
        perceptual_losses = []
        for name, block in self.vgg_blocks.items():
            features = block(features)
            if name in self.perceptual_losses:
                perceptual_losses.append(self.perceptual_losses[name](features))
            if name in self.style_losses:
                style_losses.append(self.style_losses[name](features))

        return sum(style_losses), sum(perceptual_losses)


class PerceptualAndStyleLoss(AbstractPerceptualAndStyleLoss):

    def __init__(self, use_perceptual_loss=True, use_style_loss=True):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        vgg_features = self.adapt_vgg_model(vgg.features)

        blocks = {
            'conv1_1': vgg_features[:2],
            'conv1_2': vgg_features[2:4],
            'conv2_2': vgg_features[4:9],
            'conv3_3': vgg_features[9:16],
        }
        self.vgg_blocks = nn.ModuleDict(blocks)
        self.vgg_blocks.eval()

        if use_perceptual_loss:
            self.perceptual_blocks = list(self.vgg_blocks.keys())
        else:
            self.perceptual_blocks = []

        if use_style_loss:
            self.style_blocks = ['conv3_3']
        else:
            self.style_blocks = []

    def forward(self, image: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = image
        target_features = target
        style_losses = []
        perceptual_losses = []
        for name, block in self.vgg_blocks.items():
            image_features = block(image_features)
            target_features = block(target_features)

            if mask is not None:
                mask = F.interpolate(mask, target_features.shape[-2:], mode='bilinear')

            if name in self.perceptual_blocks:
                perceptual_losses.append(self.run_loss(image_features, target_features, mask, PerceptualLoss))

            if name in self.style_blocks:
                style_losses.append(self.run_loss(image_features, target_features, mask, StyleLoss))

        return sum(style_losses), sum(perceptual_losses)

    @staticmethod
    def run_loss(image_features: torch.Tensor, target_features: torch.Tensor, mask: torch.Tensor, loss_class: Union[PerceptualLoss, StyleLoss]) -> torch.Tensor:
        loss_func = loss_class(target_features, mask=mask)
        return loss_func(image_features)


class StyleLossNetwork(AbstractPerceptualAndStyleLoss):

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        vgg_features = self.adapt_vgg_model(vgg.features)

        blocks = {
            'conv3_3': vgg_features[:16],
        }
        self.vgg_blocks = nn.ModuleDict(blocks)
        self.vgg_blocks.eval()

        self.disable_update()

    def forward(self, generated_image: torch.Tensor, style_image: torch.Tensor) -> torch.Tensor:
        generated_features = generated_image
        style_features = style_image
        style_losses = []

        for name, block in self.vgg_blocks.items():
            generated_features = block(generated_features)
            style_features = block(style_features)

            loss_func = StyleLoss(style_features)
            style_loss = loss_func(generated_features)
            style_losses.append(style_loss)

        return sum(style_losses)
