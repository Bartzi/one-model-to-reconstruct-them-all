from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Callable

import torch
from torch import optim

from latent_projecting.losses import w_plus_style_loss, noise_loss, w_plus_loss, naive_noise_loss
from pytorch_training.optimizer.lr_scheduling import LambdaLRWithRamp


@dataclass
class Latents:
    latent: torch.Tensor
    noise: List[torch.Tensor]

    def to(self, device) -> Latents:
        self.latent = self.latent.to(device)
        self.noise = [noise.to(device) for noise in self.noise]
        return self

    def __getitem__(self, key: int) -> Latents:
        latent = self.latent[key].unsqueeze(0)
        noise = [noise[key].unsqueeze(0) for noise in self.noise]
        return Latents(latent, noise)

    def detach(self):
        self.latent = self.latent.detach()
        self.noise = [noise.detach() for noise in self.noise]

    def numpy(self) -> Latents:
        latent = self.latent.cpu().numpy()
        noises = [noise.cpu().numpy() for noise in self.noise]
        return Latents(latent, noises)


@dataclass
class CodeLatents(Latents):
    code: torch.Tensor

    def to(self, device) -> CodeLatents:
        super().to(device)
        self.code = self.code.to(device)
        return self

    def __getitem__(self, key: int) -> CodeLatents:
        latent = self.latent[key].unsqueeze(0)
        noise = [noise[key].unsqueeze(0) for noise in self.noise]
        code = self.code[key].unsqueeze(0)
        return CodeLatents(latent, noise, code)

    def detach(self):
        super().detach()
        self.code = self.code.detach()


@dataclass
class LatentPaths:
    latent: List[torch.Tensor]
    noise: List[List[torch.Tensor]]

    def to(self, device) -> LatentPaths:
        for i in range(len(self)):
            self.latent[i] = self.latent[i].to(device)
            self.noise[i] = [noise.to(device) for noise in self.noise[i]]
        return self

    def __len__(self):
        assert len(self.latent) == len(self.noise)
        return len(self.latent)

    def __iter__(self) -> Latents:
        for latent, noise in zip(self.latent, self.noise):
            yield Latents(latent, noise)

    def __add__(self, other: LatentPaths) -> LatentPaths:
        self.latent += other.latent
        self.noise += other.noise
        return self

    def split(self) -> List[LatentPaths]:
        latent_paths = torch.stack(self.latent, dim=0)
        latent_paths = torch.transpose(latent_paths, 0, 1)
        latent_paths = torch.split(latent_paths, 1)
        latent_paths = [torch.split(tensor[0], 1) for tensor in latent_paths]

        noise_paths = defaultdict(list)
        for path_element in self.noise:
            splitted_noises = defaultdict(list)
            for noise_element in path_element:
                splitted_batch = noise_element.split(1)
                for i in range(len(splitted_batch)):
                    splitted_noises[i].append(splitted_batch[i])

            for batch_index, noises in splitted_noises.items():
                noise_paths[batch_index].append(noises)

        return [LatentPaths(list(latent), list(noises)) for latent, noises in zip(latent_paths, noise_paths.values())]


def optimize_noise(args: Namespace, projector: "Projector", latents: Latents, images: torch.Tensor, loss_func: Callable) -> Tuple[LatentPaths, Latents]:
    latents.to(projector.device)
    projector.set_requires_grad(latents, False)
    optimizer = optim.Adam(latents.noise, lr=args.noise_lr)
    scheduling_function = LambdaLRWithRamp.get_lr_with_ramp(args.noise_step, args.noise_lr_rampdown, args.noise_lr_rampup)
    lr_scheduler = LambdaLRWithRamp(optimizer, scheduling_function)

    paths, best_latent = projector.project(
        latents,
        images,
        optimizer,
        args.noise_step,
        loss_func,
        lr_scheduler=lr_scheduler,
    )

    return paths, best_latent


def run_image_reconstruction(args: Namespace, projector: "Projector", latents: Latents, images: torch.Tensor, do_optimize_noise: bool = True, latent_abort_condition: Callable = None, noise_abort_condition: Callable = None) -> Tuple[LatentPaths, Latents]:
    latents.to(projector.device)
    projector.abort_condition = latent_abort_condition
    projector.set_requires_grad(latents, True)
    optimizer = optim.Adam([latents.latent], lr=args.lr)
    scheduling_function = LambdaLRWithRamp.get_lr_with_ramp(args.latent_step, args.lr_rampdown, args.lr_rampup)
    lr_scheduler = LambdaLRWithRamp(optimizer, scheduling_function)

    paths, best_latent = projector.project(
        latents,
        images,
        optimizer,
        args.latent_step,
        w_plus_loss({"l_percept": 1, "l_mse": args.mse}, args.device),
        lr_scheduler=lr_scheduler,
    )

    if do_optimize_noise:
        projector.abort_condition = noise_abort_condition
        more_paths, best_latent = optimize_noise(
            args,
            projector,
            best_latent,
            images,
            naive_noise_loss({"l_mse": 1})
        )
        paths = LatentPaths(paths.latent + more_paths.latent, paths.noise + more_paths.noise)

    return paths, best_latent


def run_local_style_transfer(args: Namespace, projector: "Projector", latents: Latents, content_image: torch.Tensor, style_image: torch.Tensor, mask_image: torch.Tensor) -> Tuple[LatentPaths, Latents]:
    latents.to(projector.device)
    projector.set_requires_grad(latents, True)
    optimizer = optim.Adam([latents.latent], lr=args.lr)
    scheduling_function = LambdaLRWithRamp.get_lr_with_ramp(args.style_latent_step, args.style_lr_rampdown, args.style_lr_rampup)
    lr_scheduler = LambdaLRWithRamp(optimizer, scheduling_function)

    latent_path, best_latent = projector.project(
        latents,
        content_image,
        optimizer,
        args.style_latent_step,
        w_plus_style_loss({"l_percept": 1, "l_mse": 1, "l_style": 1}, content_image, style_image, mask_image, args.device),
        lr_scheduler=lr_scheduler,
    )

    # optimize noise
    projector.set_requires_grad(best_latent, False)
    optimizer = optim.Adam(best_latent.noise, lr=args.noise_lr)
    scheduling_function = LambdaLRWithRamp.get_lr_with_ramp(args.style_noise_step, args.noise_style_lr_rampdown, args.noise_style_lr_rampup)
    lr_scheduler = LambdaLRWithRamp(optimizer, scheduling_function)

    more_latent_path, more_noise_path = projector.project(
        best_latent.to(args.device),
        content_image,
        optimizer,
        args.style_noise_step,
        noise_loss(
            {"l_mse_1": 1, "l_mse_2": 1},
            content_image,
            projector.generate(best_latent.to(args.device))[0],
            # style_image,
            mask_image
        ),
        lr_scheduler=lr_scheduler,
    )

    latent_path = latent_path + more_latent_path

    return latent_path, best_latent
