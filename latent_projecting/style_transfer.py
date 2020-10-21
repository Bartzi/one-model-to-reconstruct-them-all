from argparse import Namespace
from pathlib import Path
from typing import Union, Tuple, Optional, Dict

import torch
from PIL import Image

from latent_projecting import Latents, run_image_reconstruction, LatentPaths, noise_loss, optimize_noise
from networks import UNetLikeEncoder
from networks.stylegan1.model import Generator as StyleGan1Generator
from networks.stylegan2.model import Generator as StyleGan2Generator
from pytorch_training.images.utils import is_image, load_and_prepare_image, clamp_and_unnormalize, make_image
from latent_projecting.projector import Projector


class StyleTransferer:

    def __init__(self, opts: Namespace) -> None:
        self.args = opts
        self.projector = Projector(opts)

    def embed_image(self, image_path: Union[str, Path], is_content_image: bool = False) -> Latents:
        image = load_and_prepare_image(image_path, self.projector.get_transforms()).to(self.projector.device)
        latents_in = self.projector.create_initial_latent_and_noise()

        _, best_latent = run_image_reconstruction(self.projector.args, self.projector, latents_in, image)

        return best_latent

    def get_latents(self, content_path: Union[str, Path], style_path: Union[str, Path]) -> Tuple[Latents, Latents]:
        if is_image(content_path):
            content_latents = self.embed_image(content_path, True)
        else:
            embedded_data = torch.load(content_path)
            content_latents = Latents(embedded_data['latent'], embedded_data['noise'])

        if is_image(style_path):
            style_latents = self.embed_image(style_path, False)
        else:
            embedded_data = torch.load(style_path)
            style_latents = Latents(embedded_data['latent'], embedded_data['noise'])

        for latents in [content_latents, style_latents]:
            if len(latents.latent.shape) < 3:
                latents.latent = latents.latent.unsqueeze(0)

        return content_latents.to(self.projector.device), style_latents.to(self.projector.device)

    def post_noise_optimize(self, content_latent: Latents, transfer_latent: Latents) -> Tuple[LatentPaths, Latents]:
        content_latent = content_latent.to(self.projector.device)
        transfer_latent = transfer_latent.to(self.projector.device)

        content_image = self.projector.generate(content_latent)[0].detach()
        style_image = self.projector.generate(transfer_latent)[0].detach()
        content_mask = clamp_and_unnormalize(content_image.clone().detach())
        loss_func = noise_loss(
            {"l_mse_1": 1, "l_mse_2": 1},
            content_image,
            style_image,
            (1 - content_mask).detach()
        )

        path, latent_and_noise = optimize_noise(self.args, self.projector, transfer_latent, content_image, loss_func)

        return path, latent_and_noise

    def do_style_transfer(self, content_latent: Latents, style_latent: Latents, layer_id: int) -> Tuple[torch.Tensor, Optional[LatentPaths]]:
        latent = torch.cat([content_latent.latent[:, :layer_id, :], style_latent.latent[:, layer_id:, :]], dim=1).detach().clone()
        latent = latent.to(self.projector.device)
        # noise = content_latent.noise[:layer_id] + style_latent.noise[layer_id:]
        noise = [n.detach().clone().to(self.projector.device) for n in content_latent.noise]
        latent_and_noise = Latents(latent, noise)

        path = None
        if self.args.post_optimize:
            path, latent_and_noise = self.post_noise_optimize(content_latent, latent_and_noise)

        latent_and_noise = latent_and_noise.to(self.projector.device)
        return self.projector.generate(latent_and_noise)[0], path

    def save_stylized_images(self, stylized_images: Dict[int, Tuple[torch.Tensor, Optional[LatentPaths]]], content_path: Path, style_path: Path, destination_dir: Path, create_gif: bool = False):
        for index, (image_array, optimization_path) in stylized_images.items():
            content_name = content_path.stem
            style_name = style_path.stem

            image_name = f"{content_name}_{style_name}_{index}"
            destination_name = destination_dir / f"{image_name}.png"
            Image.fromarray(make_image(image_array)[0]).save(destination_name)

            if optimization_path is not None and create_gif:
                self.projector.create_gif(optimization_path, image_name, destination_dir)


class EncoderBasedStyleTransferer(StyleTransferer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_encoder = self.build_encoder(self.args.content_checkpoint)
        self.style_encoder = self.build_encoder(self.args.style_checkpoint)

    def build_encoder(self, checkpoint) -> UNetLikeEncoder:
        if self.projector.config['stylegan_variant'] == 1:
            channel_map = StyleGan1Generator.get_channels()
        else:
            channel_map = StyleGan2Generator.get_channels()

        encoder = UNetLikeEncoder(
            self.projector.config['image_size'],
            self.projector.config['latent_size'],
            self.projector.config['input_dim'],
            channel_map
        )
        encoder.eval()

        checkpoint = torch.load(checkpoint)

        if 'autoencoder' in checkpoint:
            # we need to adapt the tensors we actually want to load
            stripped_checkpoint = {key: value for key, value in checkpoint['autoencoder'].items() if 'encoder' in key}
            checkpoint = {'.'.join(key.split('.')[2:]): value for key, value in stripped_checkpoint.items()}

        encoder.load_state_dict(checkpoint)

        return encoder.to(self.projector.device)

    def embed_image(self, image_path: Union[str, Path], is_content_image: bool = True) -> Latents:
        image = load_and_prepare_image(image_path, self.projector.get_transforms()).to(self.projector.device)
        if is_content_image:
            encoder = self.content_encoder
        else:
            encoder = self.style_encoder

        with torch.no_grad():
            return encoder(image)
