import argparse
import functools
import shutil
from pathlib import Path
from typing import Union, Callable

import torch
import warnings
from torch import nn

from networks.encoder.autoencoder import StyleganAutoencoder, CodeStyleganAutoencoder, \
    ContentAndStyleStyleganAutoencoder, TwoStemStyleganAutoencoder, DropoutStyleganAutoencoder, \
    SuperResolutionStyleganAutoencoder
from networks.encoder.resnet_based_encoder import Encoder
from networks.encoder.u_net_like_encoder import UNetLikeEncoder, WPlusEncoder, WEncoder, WWPlusEncoder, WCodeEncoder, \
    WPlusResnetNoiseEncoder, WPlusNoNoiseEncoder, NoiseEncoder, WNoNoiseEncoder
from networks.stylegan1.model import StyledGenerator, Generator as StyleGan1Generator
from utils.convert_autoencoder_checkpoint import convert_autoencoder_checkpoint


def load_weights(network: nn.Module, model_file: Union[str, Path], *, key: str = None, strict: bool = True, convert: bool = False) -> nn.Module:
    weights = torch.load(model_file)
    if convert:
        weights = convert_autoencoder_checkpoint(weights)
    if key is not None and key in weights:
        weights = weights[key]
    network.load_state_dict(weights, strict=strict)
    return network


if shutil.which('ninja'):
    from networks.stylegan2.model import Generator


    def get_stylegan2_generator(image_size, latent_size, n_mlp=8, channel_multiplier=2, init_ckpt=None, ckpt_key='g_ema', strict=True) -> Generator:
        generator = Generator(image_size, latent_size, n_mlp, channel_multiplier=channel_multiplier)
        if init_ckpt is not None:
            load_weights(generator, init_ckpt, key=ckpt_key, strict=strict)

        return generator


    def get_stylegan2_resnet_autoencoder(image_size, latent_size, num_input_channels, n_mlp=8, channel_multiplier=2, init_ckpt=None) -> nn.Module:
        generator = get_stylegan2_generator(
            image_size,
            latent_size,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
            init_ckpt=init_ckpt,
            strict=False,
        )

        encoder = Encoder(
            image_size,
            latent_size,
            num_input_channels,
            generator.channels,
        )

        autoencoder = StyleganAutoencoder(encoder, generator)
        return autoencoder


    def get_stylegan2_autoencoder(image_size, latent_size, num_input_channels, n_mlp=8, channel_multiplier=2, init_ckpt=None, encoder_class: Union[WPlusEncoder, WWPlusEncoder] = WPlusEncoder, autoencoder_class: Union[StyleganAutoencoder, DropoutStyleganAutoencoder] = StyleganAutoencoder) -> StyleganAutoencoder:
        generator = get_stylegan2_generator(
            image_size,
            latent_size,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
            init_ckpt=init_ckpt,
            strict=False
        )

        encoder = encoder_class(
            image_size,
            latent_size,
            num_input_channels,
            generator.channels,
            stylegan_variant=2,
        )

        autoencoder = autoencoder_class(encoder, generator)
        return autoencoder

    def get_stylegan2_wplus_style_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None) -> ContentAndStyleStyleganAutoencoder:
        generator = get_stylegan2_generator(
            image_size,
            latent_size,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
            init_ckpt=init_ckpt
        )

        encoder = WPlusEncoder(
            image_size,
            latent_size,
            num_input_channels,
            StyleGan1Generator.get_channels(),
            stylegan_variant=1
        )

        autoencoder = ContentAndStyleStyleganAutoencoder(encoder, generator)
        return autoencoder

    def get_stylegan_2_two_stem_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None, update_latent=True, update_noise=True, encoder_class : Union[WPlusNoNoiseEncoder, WNoNoiseEncoder] = WPlusNoNoiseEncoder) -> TwoStemStyleganAutoencoder:
        generator = get_stylegan2_generator(
            image_size,
            latent_size,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
            init_ckpt=init_ckpt,
            strict=False,
        )

        latent_encoder = encoder_class(
            image_size,
            latent_size,
            num_input_channels,
            StyleGan1Generator.get_channels(),
            stylegan_variant=2
        )

        noise_encoder = NoiseEncoder(
            image_size,
            latent_size,
            num_input_channels,
            StyleGan1Generator.get_channels(),
            stylegan_variant=2
        )

        autoencoder = TwoStemStyleganAutoencoder(
            latent_encoder,
            noise_encoder,
            generator,
            update_latent=update_latent,
            update_noise=update_noise
        )
        return autoencoder


def get_stylegan1_generator(image_size, latent_size, n_mlp=8, init_ckpt=None, ckpt_key='g_running', strict=True) -> StyledGenerator:
    generator = StyledGenerator(image_size, latent_size, n_mlp=n_mlp)

    if init_ckpt is not None:
        load_weights(generator, init_ckpt, key=ckpt_key, strict=strict)

    return generator


def get_stylegan1_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None, autoencoder_class=StyleganAutoencoder, encoder_class: Union[WPlusEncoder, WWPlusEncoder] = WPlusEncoder) -> StyleganAutoencoder:
    generator = get_stylegan1_generator(
        image_size,
        latent_size,
        n_mlp=n_mlp,
        init_ckpt=init_ckpt
    )

    encoder = encoder_class(
        image_size,
        latent_size,
        num_input_channels,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    autoencoder = autoencoder_class(encoder, generator)
    return autoencoder


def get_stylegan1_wplus_noise_renset_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None) -> StyleganAutoencoder:
    generator = get_stylegan1_generator(
        image_size,
        latent_size,
        n_mlp=n_mlp,
        init_ckpt=init_ckpt
    )

    encoder = WPlusResnetNoiseEncoder(
        image_size,
        latent_size,
        num_input_channels,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    autoencoder = StyleganAutoencoder(encoder, generator)
    return autoencoder


def get_stylegan1_wplus_style_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None) -> ContentAndStyleStyleganAutoencoder:
    generator = get_stylegan1_generator(
        image_size,
        latent_size,
        n_mlp=n_mlp,
        init_ckpt=init_ckpt
    )

    encoder = WPlusEncoder(
        image_size,
        latent_size,
        num_input_channels * 2,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    autoencoder = ContentAndStyleStyleganAutoencoder(encoder, generator)
    return autoencoder


def get_stylegan1_code_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None, code_dim: int = 10) -> CodeStyleganAutoencoder:
    generator = get_stylegan1_generator(
        image_size,
        latent_size + code_dim,
        n_mlp=n_mlp,
        init_ckpt=init_ckpt
    )

    encoder = WCodeEncoder(
        code_dim,
        image_size,
        latent_size,
        num_input_channels,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    autoencoder = CodeStyleganAutoencoder(encoder, generator)
    return autoencoder


def get_stylegan_1_two_stem_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None, update_latent=True, update_noise=True, encoder_class : Union[WPlusNoNoiseEncoder, WNoNoiseEncoder] = WPlusNoNoiseEncoder) -> TwoStemStyleganAutoencoder:
    generator = get_stylegan1_generator(
        image_size,
        latent_size,
        n_mlp=n_mlp,
        init_ckpt=init_ckpt
    )

    latent_encoder = encoder_class(
        image_size,
        latent_size,
        num_input_channels,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    noise_encoder = NoiseEncoder(
        image_size,
        latent_size,
        num_input_channels,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    autoencoder = TwoStemStyleganAutoencoder(
        latent_encoder,
        noise_encoder,
        generator,
        update_latent=update_latent,
        update_noise=update_noise
    )
    return autoencoder


def get_stylegan_1_superresolution_autoencoder(image_size: int, latent_size: int, num_input_channels: int, n_mlp: int = 8, channel_multiplier: int = 2, init_ckpt: str = None, input_size: int = None, encoder_class: Union[WPlusEncoder, WWPlusEncoder] = WPlusEncoder, autoencoder_kwargs: dict = None) -> SuperResolutionStyleganAutoencoder:
    if input_size is None:
        input_size = image_size
        warnings.warn("You wanted to train superresolution but you did not supply a new output size")

    assert input_size <= image_size, "For training superresolution, the image size must be greater or equal than the input size"

    generator = get_stylegan1_generator(
        image_size,
        latent_size,
        n_mlp=n_mlp,
        init_ckpt=init_ckpt
    )

    encoder = encoder_class(
        input_size,
        latent_size,
        num_input_channels,
        StyleGan1Generator.get_channels(),
        stylegan_variant=1
    )

    autoencoder = SuperResolutionStyleganAutoencoder(encoder, generator, **autoencoder_kwargs)
    return autoencoder


def get_stylegan_1_based_autoencoder(args: argparse.Namespace) -> Callable:
    if getattr(args, 'two_stem', False):
        update_latent = args.disable_update_for in ['noise', 'none']
        update_noise = args.disable_update_for in ['latent', 'none']
        encoder_class = WNoNoiseEncoder if args.w_only else WPlusNoNoiseEncoder
        autoencoder_func = functools.partial(
            get_stylegan_1_two_stem_autoencoder,
            update_latent=update_latent,
            update_noise=update_noise,
            encoder_class=encoder_class,
        )
        return autoencoder_func

    encoder_class = WWPlusEncoder if args.w_only else WPlusEncoder
    if args.code_dim > 0:
        autoencoder_func = functools.partial(get_stylegan1_code_autoencoder, code_dim=args.code_dim)
    elif getattr(args, 'superresolution', False):
        autoencoder_func = functools.partial(
            get_stylegan_1_superresolution_autoencoder,
            encoder_class=encoder_class,
            input_size=args.downsample_size,
            autoencoder_kwargs={"extend_noise_with_random": args.extend_noise_with_random},
        )
    else:
        autoencoder_func = functools.partial(get_stylegan1_autoencoder, encoder_class=encoder_class)

    if getattr(args, 'dropout_autoencoder', False):
        autoencoder_func = functools.partial(autoencoder_func, autoencoder_class=DropoutStyleganAutoencoder)

    return autoencoder_func


def get_stylegan_2_based_autoencoder(args: argparse.Namespace) -> Callable:
    try:
        from networks import get_stylegan2_autoencoder
    except ImportError:
        raise RuntimeError("stylegan 2 not supported on fsoc lab")

    if getattr(args, 'two_stem', False):
        update_latent = args.disable_update_for in ['noise', 'none']
        update_noise = args.disable_update_for in ['latent', 'none']
        encoder_class = WNoNoiseEncoder if args.w_only else WPlusNoNoiseEncoder
        autoencoder_func = functools.partial(
            get_stylegan_2_two_stem_autoencoder,
            update_latent=update_latent,
            update_noise=update_noise,
            encoder_class=encoder_class,
        )
        return autoencoder_func

    encoder_class = WWPlusEncoder if args.w_only else WPlusEncoder
    if args.code_dim > 0:
        raise NotImplementedError("stylegan2 code dim training not yet implemented")
    elif getattr(args, 'superresolution', False):
        autoencoder_func = functools.partial(
            get_stylegan_2_superresolution_autoencoder,
            encoder_class=encoder_class,
            input_size=args.downsample_size,
            autoencoder_kwargs={"extend_noise_with_random": args.extend_noise_with_random},
        )
    else:
        autoencoder_func = functools.partial(get_stylegan2_autoencoder, encoder_class=encoder_class)

    if getattr(args, 'dropout_autoencoder', False):
        autoencoder_func = functools.partial(autoencoder_func, autoencoder_class=DropoutStyleganAutoencoder)

    return autoencoder_func


def get_autoencoder(config: dict, init_ckpt: str = None) -> Union[StyleganAutoencoder, TwoStemStyleganAutoencoder]:
    assert config['stylegan_variant'] in [1, 2], "Stylegan Variant Unknown"

    if config['stylegan_variant'] == 1:
        autoencoder_func = get_stylegan_1_based_autoencoder(argparse.Namespace(**config))
    else:
        autoencoder_func = get_stylegan_2_based_autoencoder(argparse.Namespace(**config))

    autoencoder = autoencoder_func(
        config['image_size'],
        config['latent_size'],
        config['input_dim'],
        init_ckpt=init_ckpt,
    )
    return autoencoder
