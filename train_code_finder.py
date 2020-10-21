import argparse
import datetime
import functools
import math
import multiprocessing
import os
from typing import Union

import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataset_class
from evaluation.autoencoder_evaluation import AutoEncoderEvalFunc
from extensions.fid_score import FIDScore
from networks import load_weights, get_autoencoder
from networks.stylegan1.model import Discriminator as Stylegan1Discriminator
from networks.stylegan2.model import Discriminator as Stylegan2Discriminator
from pytorch_training.distributed import get_rank, get_world_size, synchronize
from pytorch_training.extensions import ImagePlotter, Snapshotter, Evaluator
from pytorch_training.extensions.logger import WandBLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.optimizer import GradientClipAdam
from pytorch_training.trainer import DistributedTrainer
from pytorch_training.triggers import get_trigger
from updater.autoencoder_discriminator_updater import AutoencoderDiscriminatorUpdater
from updater.autoencoder_updater import AutoencoderUpdater
from utils.config import load_yaml_config
from utils.data_loading import build_data_loader


def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    for key in dir(args):
        if key.startswith("_"):
            continue
        config[key] = getattr(args, key)
    return config


def get_discriminator(config: dict) -> Union[Stylegan1Discriminator, Stylegan2Discriminator]:
    if config['stylegan_variant'] == 1:
        discriminator = Stylegan1Discriminator(from_rgb_activate=True)
        discriminator.forward = functools.partial(discriminator.forward, step=int(math.log2(config['image_size'])) - 2, alpha=0)
    else:
        discriminator = Stylegan2Discriminator(config['image_size'])
    return discriminator


def main(args, rank, world_size):
    config = load_yaml_config(args.config)
    config = merge_config_and_args(config, args)

    dataset_class = get_dataset_class(args)
    train_data_loader = build_data_loader(args.images, config, args.absolute, dataset_class=dataset_class)

    autoencoder = get_autoencoder(config, init_ckpt=args.stylegan_checkpoint)
    discriminator = None
    if args.use_discriminator:
        discriminator = get_discriminator(config)

    if args.disable_update_for == 'latent':
        assert args.autoencoder is not None, "if you want to only train noise, we need an autoencoder checkoint!"
        print(f"Loading encoder weights from {args.autoencoder} for noise-only training.")
        load_weights(autoencoder.encoder, args.autoencoder, key='encoder', strict=False)
    elif args.autoencoder is not None:
        print(f"Loading all weights from {args.autoencoder}.")
        load_weights(autoencoder, args.autoencoder, key='autoencoder')

    optimizer_opts = {
        'betas': (config['beta1'], config['beta2']),
        'weight_decay': config['weight_decay'],
        'lr': float(config['lr']),
    }

    if args.disable_update_for != 'none':
        if float(config['lr_to_noise']) != float(config['lr']):
            print("Warning: updates for some parts of the networks are disabled. "
                  f"Therefore 'lr_to_noise'={config['lr_to_noise']} is ignored.")
        optimizer = GradientClipAdam(autoencoder.trainable_parameters(), **optimizer_opts)
    else:
        main_param_group, noise_param_group = autoencoder.trainable_parameters(
            as_groups=(["to_noise", "intermediate_to_noise"],)
        )
        noise_param_group['lr'] = float(config['lr_to_noise'])
        optimizer = GradientClipAdam([main_param_group, noise_param_group], **optimizer_opts)

    if world_size > 1:
        distributed = functools.partial(DDP, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False, output_device=rank)
        autoencoder = distributed(autoencoder.to('cuda'))
        if discriminator is not None:
            discriminator = distributed(discriminator.to('cuda'))
    else:
        autoencoder = autoencoder.to('cuda')
        if discriminator is not None:
            discriminator = discriminator.to('cuda')

    if discriminator is not None:
        discriminator_optimizer = GradientClipAdam(discriminator.parameters(), **optimizer_opts)
        updater = AutoencoderDiscriminatorUpdater(
            iterators={'images': train_data_loader},
            networks={'autoencoder': autoencoder, 'discriminator': discriminator},
            optimizers={'main': optimizer, 'discriminator': discriminator_optimizer},
            device='cuda',
            copy_to_device=world_size == 1,
            disable_update_for=args.disable_update_for,
        )
    else:
        updater = AutoencoderUpdater(
            iterators={'images': train_data_loader},
            networks={'autoencoder': autoencoder},
            optimizers={'main': optimizer},
            device='cuda',
            copy_to_device=world_size == 1,
            disable_update_for=args.disable_update_for,
        )

    trainer = DistributedTrainer(
        updater,
        stop_trigger=get_trigger((config['max_iter'], 'iteration'))
    )

    logger = WandBLogger(
        args.log_dir,
        args,
        config,
        os.path.dirname(os.path.realpath(__file__)),
        trigger=get_trigger((config['log_iter'], 'iteration')),
        master=rank == 0,
        project_name="One Model to Generate them All",
        run_name=args.log_name,
    )

    if args.val_images is not None:
        val_data_loader = build_data_loader(args.val_images, config, args.absolute, shuffle_off=True, dataset_class=dataset_class)

        evaluator = Evaluator(
            val_data_loader,
            logger,
            AutoEncoderEvalFunc(autoencoder, rank),
            rank,
            trigger=get_trigger((1, 'epoch'))
        )
        trainer.extend(evaluator)

    fid_extension = FIDScore(
        autoencoder if not isinstance(autoencoder, DDP) else autoencoder.module,
        val_data_loader if args.val_images is not None else train_data_loader,
        dataset_path=args.val_images if args.val_images is not None else args.images,
        trigger=(1, 'epoch')
    )
    trainer.extend(fid_extension)

    if rank == 0:
        snapshot_autoencoder = autoencoder if not isinstance(autoencoder, DDP) else autoencoder.module
        snapshotter = Snapshotter(
            {
                'autoencoder': snapshot_autoencoder,
                'encoder': snapshot_autoencoder.encoder,
                'decoder': snapshot_autoencoder.decoder,
                'optimizer': optimizer,
            },
            args.log_dir,
            trigger=get_trigger((config['snapshot_save_iter'], 'iteration'))
        )
        trainer.extend(snapshotter)

        plot_images = []
        if args.val_images is not None:
            def fill_plot_images(data_loader):
                image_list = []
                num_images = 0
                for batch in data_loader:
                    for image in batch['input_image']:
                        image_list.append(image)
                        num_images += 1
                        if num_images > config['display_size']:
                            return image_list
                raise RuntimeError(f"Could not gather enough plot images for display size {config['display_size']}.")

            plot_images = fill_plot_images(val_data_loader)
        else:
            for i in range(config['display_size']):
                if hasattr(train_data_loader.sampler, 'set_epoch'):
                    train_data_loader.sampler.set_epoch(i)
                plot_images.append(next(iter(train_data_loader))['input_image'][0])
        image_plotter = ImagePlotter(plot_images, [autoencoder], args.log_dir, trigger=get_trigger((config['image_save_iter'], 'iteration')), plot_to_logger=True)
        trainer.extend(image_plotter)

    schedulers = {
        "encoder": CosineAnnealingLR(optimizer, config["max_iter"], eta_min=1e-8)
    }
    lr_scheduler = LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))
    trainer.extend(lr_scheduler)

    trainer.extend(logger)

    synchronize()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model that predicts a stylegan latent code through autoencoding",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="path to yaml config to use for training")
    parser.add_argument("stylegan_checkpoint", help='Path to saved stylegan checkpoint that is used for optimization')
    parser.add_argument("--images", required=True, help="path to json file holding a list of all images to use")
    parser.add_argument("--val-images", help="path to json holding validation images (same data format as train images)")
    parser.add_argument("-s", "--stylegan-variant", type=int, choices=[1, 2], default=2, help="which stylegan variant to use")
    parser.add_argument("--absolute", action='store_true', default=False, help="indicate that your json contains absolute paths")
    parser.add_argument('-l', '--log-dir', default='training', help="outputs path")
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mpi-backend', default='gloo', choices=['nccl', 'gloo'], help="MPI backend to use for interprocess communication")
    parser.add_argument("-w", "--w-only", action='store_true', default=False, help="embed only in W space, not W+")
    parser.add_argument("-c", "--code-dim", type=int, default=0, help="train info stylegan like net with code dim (use a model pretrained for info style gan)")
    parser.add_argument("-d", "--disable-update-for", choices=('noise', 'latent', 'none'), default='none', help="indicate that you want to disable update noise or latent part of encoder during training")
    parser.add_argument("--autoencoder", help="path to pretrained autoencoder that is used to train noise")
    parser.add_argument("--dropout-autoencoder", action='store_true', default=False, help="use dropout autoencoder")
    parser.add_argument("--two-stem", action='store_true', default=False, help="train two stem network")
    parser.add_argument("--use-discriminator", action='store_true', default=False, help="Train autoencoder with extra Discriminator as loss")
    parser.add_argument("--superresolution", action='store_true', default=False, help="Train autoencoder for superresolution")
    parser.add_argument("--denoising", action='store_true', default=False, help="Train autoencoder for image denoising")
    parser.add_argument("--black-and-white-denoising", action='store_true', default=False, help="Train autoencoder for black and white denoising")

    args = parser.parse_args()
    args.log_dir = os.path.join('logs', args.log_dir, args.log_name, datetime.datetime.now().isoformat())

    if torch.cuda.device_count() > 1:
        multiprocessing.set_start_method('forkserver')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.mpi_backend, init_method='env://')
        synchronize()

    main(args, get_rank(), get_world_size())
