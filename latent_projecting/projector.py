import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms
from tqdm import tqdm

from latent_projecting import Latents, LatentPaths
from utils.config import load_config
from losses.psnr import PSNR
from networks import get_stylegan1_generator, StyledGenerator, get_stylegan2_generator
from networks.stylegan2.model import Generator
from pytorch_training.data import Compose
from pytorch_training.images.utils import make_image
from utils.image_utils import render_text_on_image


class Projector:

    def __init__(self, args, abort_condition=None):
        self.args = args
        self.abort_condition = abort_condition

        self.device = args.device
        self.config = load_config(args.ckpt, args.config)
        self.generator = self.load_generator()
        self.generator.eval()
        self.debug_step = args.debug_step

        self.psnr = PSNR()
        self.log = []

    def reset(self):
        self.log.clear()

    def load_generator(self) -> Union[Generator, StyledGenerator]:
        if self.config['stylegan_variant'] == 2:
            generator = get_stylegan2_generator(self.config['image_size'], self.config['latent_size'],
                                                init_ckpt=self.config['stylegan_checkpoint'])
        else:
            generator = get_stylegan1_generator(self.config['image_size'], self.config['latent_size'],
                                                init_ckpt=self.config['stylegan_checkpoint'])

        generator.eval()
        generator = generator.to(self.device)
        return generator

    def get_transforms(self) -> Compose:
        return Compose(
            [
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def get_blur_transform(self, from_tensor: bool = True) -> list:
        blur_transform = [transforms.Lambda(lambda image: image.filter(ImageFilter.GaussianBlur(radius=3)))]
        if from_tensor:
            blur_transform.insert(0, transforms.ToPILImage())
            blur_transform.append(transforms.ToTensor())
        return blur_transform

    def get_mask_transform(self, invert_mask: bool = False, mask_multiplier: float = 1) -> Compose:
        transformations = [
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
        if invert_mask:
            transformations.append(transforms.Lambda(lambda image: 1 - image))

        if mask_multiplier < 1:
            multiplier_transform = transforms.Lambda(lambda image: image * mask_multiplier)
            transformations.append(multiplier_transform)

        transformations.extend(self.get_blur_transform())

        return Compose(transformations)

    @staticmethod
    def sample_mean_latent(sample_size: int, latent_size: int, device: str, generator: Union[StyledGenerator, Generator]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            noise_sample = torch.randn(sample_size, latent_size, device=device)
            latent_out = generator.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / sample_size) ** 0.5
        return latent_mean, latent_std

    def get_mean_latent(self, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_mean_latent(sample_size, self.config['latent_size'], self.device, self.generator)

    def set_requires_grad(self, latents: Latents, flag: bool):
        latents.latent.requires_grad = flag

        for noise in latents.noise:
            noise.requires_grad = not flag

    def create_initial_latent_and_noise(self) -> Latents:
        n_mean_latent = 10000
        latent_mean, latent_std = self.get_mean_latent(n_mean_latent)

        base_noises = self.generator.make_noise()
        noises = [noise.detach().clone() for noise in base_noises]

        if self.args.no_mean_latent:
            latent_in = torch.normal(0, latent_std.item(), size=(1, self.config['latent_size']),
                                     device=self.device)
        else:
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1)

        if self.args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, self.generator.n_latent, 1)

        return Latents(latent_in, noises)

    def generate(self, latents: Latents) -> torch.Tensor:
        return self.generator([latents.latent], input_is_latent=True, noise=latents.noise)

    def project(self, latents: Latents, images: torch.Tensor, optimizer: Optimizer, num_steps: int, loss_function: Callable, lr_scheduler: _LRScheduler = None) -> Tuple[LatentPaths, Latents]:
        pbar = tqdm(range(num_steps), leave=False)
        latent_path = []
        noise_path = []

        best_latent = best_noise = best_psnr = None

        for i in pbar:
            img_gen, _ = self.generate(latents)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            # # n_loss = noise_regularize(noises)
            loss, loss_dict = loss_function(img_gen, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_dict['psnr'] = self.psnr(img_gen, images).item()
            loss_dict['lr'] = optimizer.param_groups[0]["lr"]

            if lr_scheduler is not None:
                lr_scheduler.step()

            self.log.append(loss_dict)

            if best_psnr is None or best_psnr < loss_dict['psnr']:
                best_psnr = loss_dict['psnr']
                best_latent = latents.latent.detach().clone().cpu()
                best_noise = [noise.detach().clone().cpu() for noise in latents.noise]

            if i % self.debug_step == 0:
                latent_path.append(latents.latent.detach().clone().cpu())
                noise_path.append([noise.detach().clone().cpu() for noise in latents.noise])

            loss_description = "; ".join(f"{key}: {value:.6f}" for key, value in loss_dict.items())
            pbar.set_description(loss_description)

            loss_dict['iteration'] = i
            if self.abort_condition is not None and self.abort_condition(loss_dict):
                break

        latent_path.append(latents.latent.detach().clone().cpu())
        noise_path.append([noise.detach().clone().cpu() for noise in latents.noise])

        return LatentPaths(latent_path, noise_path), Latents(best_latent, best_noise)

    def create_gif(self, latent_paths: LatentPaths, image_name: str, destination_dir: Path) -> None:
        destination_dir = destination_dir / 'gifs'
        destination_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(mode='w') as temp_file:
            latent_paths = latent_paths.to(self.device)
            temp_dir = Path(temp_dir)

            progress_bar = tqdm(latent_paths)
            progress_bar.set_description("creating gif")
            for i, latent in enumerate(progress_bar):
                img_gen, _ = self.generator([latent.latent], input_is_latent=True, noise=latent.noise)

                image_index = i * self.debug_step
                temp_dest_name = temp_dir / f"{image_index}.png"
                img_ar = make_image(img_gen)[0]
                image = Image.fromarray(img_ar)
                image = render_text_on_image(f"{image_index:06}", image)
                image.save(temp_dest_name)

                print(temp_dest_name, file=temp_file)

            process_args = [
                'convert',
                '-delay 10',
                '-loop 0',
                f'@{temp_file.name}',
                str(destination_dir / f"{image_name}.gif")
            ]
            temp_file.flush()
            subprocess.run(' '.join(process_args), shell=True, check=True)

    def render_log(self, destination_dir: Union[str, Path], image_base_name: str) -> None:
        destination_dir = Path(destination_dir)
        destination_dir = destination_dir / 'log'
        os.makedirs(destination_dir, exist_ok=True)

        plot_data = defaultdict(list)
        for logged_data in self.log:
            for key, value in logged_data.items():
                plot_data[key].append(value)

        for key in plot_data.keys():
            plt.clf()
            fig, axis = plt.subplots()
            axis.plot(plot_data[key])
            axis.set(ylabel=key)
            axis.grid()

            fig.savefig(destination_dir / f"{image_base_name}_{key}.png")
            plt.close(fig)

        with open(destination_dir / f"{image_base_name}_log.json", 'w') as f:
            json.dump(self.log, f, indent='\t')
