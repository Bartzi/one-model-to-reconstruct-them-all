from itertools import chain
from typing import Iterator, Sequence, List, Dict

import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from latent_projecting import Latents, CodeLatents


class StyleganAutoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_generated_noise = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_codes = self.encode(x)

        if not self.use_generated_noise:
            latent_codes.noise = self.decoder.make_noise()

        reconstructed_x, _ = self.decoder([latent_codes.latent], input_is_latent=self.is_wplus(latent_codes), noise=latent_codes.noise)
        return reconstructed_x

    def is_wplus(self, latents: Latents):
        return len(latents.latent.shape) == 3

    def trainable_parameters(self, recurse: bool = ..., as_groups: Sequence[Sequence[str]] = None) -> [Iterator[Parameter], List[Dict[str, list]]]:
        if as_groups is None:
            return self.encoder.parameters(recurse=recurse)

        main_params = []
        filtered_params = [[] for _ in as_groups]
        for name, param in self.encoder.named_parameters(recurse=recurse):
            in_group = False
            for i, key_list in enumerate(as_groups):
                if any(key in name for key in key_list):
                    filtered_params[i].append(param)
                    in_group = True
                    break
            if not in_group:
                main_params.append(param)

        return [{'params': params} for params in [main_params] + filtered_params]

    def encode(self, x: torch.Tensor) -> Latents:
        return self.encoder(x)


class DropoutStyleganAutoencoder(StyleganAutoencoder):

    def __init__(self, *args, dropout_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_ratio = dropout_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_codes = self.encode(x)

        random_noises = self.decoder.make_noise()
        mixed_noise = [predicted_noise if random.random() > self.dropout_ratio else generated_noise for predicted_noise, generated_noise in zip(latent_codes.noise, random_noises)]

        reconstructed_x, _ = self.decoder([latent_codes.latent], input_is_latent=self.is_wplus(latent_codes), noise=mixed_noise)
        return reconstructed_x


class CodeStyleganAutoencoder(StyleganAutoencoder):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_info_codes = self.encode(x)

        latent = torch.cat([latent_info_codes.latent, latent_info_codes.code], dim=1)
        reconstructed_x, _ = self.decoder([latent], input_is_latent=False, noise=latent_info_codes.noise)

        return reconstructed_x

    def encode(self, x: torch.Tensor) -> CodeLatents:
        return self.encoder(x)


class ContentAndStyleStyleganAutoencoder(StyleganAutoencoder):

    def forward(self, content_images: torch.Tensor, style_images: torch.Tensor) -> torch.Tensor:
        encoder_input_image = torch.cat([content_images, style_images], dim=1)
        latents = self.encode(encoder_input_image)

        reconstructed_x, _ = self.decoder([latents.latent], input_is_latent=self.is_wplus(latents), noise=latents.noise)
        return reconstructed_x


class SuperResolutionStyleganAutoencoder(StyleganAutoencoder):

    def __init__(self, *args, extend_noise_with_random: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.extend_noise_with_random = extend_noise_with_random

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (self.encoder.image_size, self.encoder.image_size), mode='area').detach()
        latents = self.encode(x)

        if self.decoder.size > self.encoder.image_size:
            # we have to add some noise to perform super resolution
            noises = latents.noise
            num_predicted_noise_maps = len(latents.noise)

            random_noises = self.decoder.make_noise()
            if self.extend_noise_with_random:
                noises.extend(random_noises[num_predicted_noise_maps:])
            else:
                noise_maps_to_add = len(random_noises) - num_predicted_noise_maps
                current_noise_map = noises[-1]
                for i in range(noise_maps_to_add):
                    current_noise_map = F.interpolate(
                        current_noise_map.clone().detach(),
                        random_noises[num_predicted_noise_maps + i].shape[-2:],
                        mode='bilinear'
                    )
                    noises.append(current_noise_map)

            latents.noise = noises

            # we also have to add some latent code parts if we have a w_plus latent
            if self.is_wplus(latents):
                target_num_latents = self.decoder.n_latent
                last_latent = latents.latent[:, -1, ...].unsqueeze(1).detach()
                padded_latent = last_latent.repeat((1, target_num_latents - latents.latent.shape[1], 1))
                latents.latent = torch.cat([latents.latent, padded_latent], dim=1)

        reconstructed_x, _ = self.decoder([latents.latent], input_is_latent=self.is_wplus(latents), noise=latents.noise)
        return reconstructed_x


class TwoStemStyleganAutoencoder(nn.Module):

    def __init__(self, latent_encoder, noise_encoder, decoder, update_latent=True, update_noise=True):
        super().__init__()
        self.latent_encoder = latent_encoder
        self.noise_encoder = noise_encoder
        self.decoder = decoder
        self.update_latent = update_latent
        self.update_noise = update_noise

        assert update_latent or update_noise, "'update_latent' or 'update_noise' must be true for Two Stem Autoencoder"

    @property
    def encoder(self):
        return self.latent_encoder

    def is_wplus(self, latents: Latents):
        return len(latents.latent.shape) == 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)

        reconstructed_x, _ = self.decoder([encoded.latent], input_is_latent=self.is_wplus(encoded), noise=encoded.noise)
        return reconstructed_x

    def trainable_parameters(self, recurse: bool = ..., as_groups: Sequence[Sequence[str]] = None) -> [Iterator[Parameter], List[Dict[str, list]]]:
        networks = []
        if self.update_latent:
            networks.append(self.latent_encoder)
        if self.update_noise:
            networks.append(self.noise_encoder)

        if as_groups is None:
            return chain.from_iterable([network.parameters(recurse=recurse) for network in networks])

        main_params = []
        filtered_params = [[] for _ in as_groups]
        for network in networks:
            for name, param in network.named_parameters(recurse=recurse):
                in_group = False
                for i, key_list in enumerate(as_groups):
                    if any(key in name for key in key_list):
                        filtered_params[i].append(param)
                        in_group = True
                        break
                if not in_group:
                    main_params.append(param)

        return [{'params': params} for params in [main_params] + filtered_params]

    def encode(self, x: torch.Tensor) -> Latents:
        with torch.set_grad_enabled(self.update_latent):
            latent_codes = self.latent_encoder(x).latent

        if self.update_noise:
            noise_codes = self.noise_encoder(x).noise
        else:
            noise_codes = self.decoder.make_noise()

        return Latents(latent=latent_codes, noise=noise_codes)
