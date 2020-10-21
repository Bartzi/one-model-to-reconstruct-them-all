import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm.contrib import tenumerate

from data.denoising_eval_dataset import DenoisingEvaluationDataset
from evaluation.psnr_ssim import PSNRSSIMEvaluator
from networks import get_autoencoder, load_weights
from pytorch_training.images.utils import clamp_and_unnormalize, make_image
from utils.config import load_config
from utils.data_loading import build_data_loader


def save_images(images: List[torch.Tensor], save_dir: Path, index: int):
    dest_file_name = save_dir / f"{index}.png"

    images = [Image.fromarray(make_image(image, normalize_func=lambda x: x)) for image in images]

    dest_image = Image.new((im := images[0]).mode, (im.width * len(images), im.height))
    for i, image in enumerate(images):
        dest_image.paste(image, (image.width * i, 0))

    dest_image.save(str(dest_file_name))


def evaluate_denoising(args):
    config = load_config(args.model_checkpoint, None)
    args.test_dataset = Path(args.test_dataset)

    assert config['denoising'] is True or config['black_and_white_denoising'] is True, "you are supplying a train run that has not been trained for denoising! Aborting"

    autoencoder = get_autoencoder(config).to(args.device)
    autoencoder = load_weights(autoencoder, args.model_checkpoint, key='autoencoder', strict=True)

    config['batch_size'] = 1
    data_loader = build_data_loader(args.test_dataset, config, config['absolute'], shuffle_off=True, dataset_class=DenoisingEvaluationDataset)

    metrics = defaultdict(list)
    psnr_ssim_evaluator = PSNRSSIMEvaluator()

    train_run_root_dir = Path(args.model_checkpoint).parent.parent
    evaluation_root = train_run_root_dir / 'evaluation' / f"denoise_{args.dataset_name}"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    for i, batch in tenumerate(data_loader, leave=False):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            denoised = autoencoder(batch['noisy'])

        noisy = clamp_and_unnormalize(batch['noisy'])
        original = clamp_and_unnormalize(batch['original'])
        denoised = clamp_and_unnormalize(denoised)

        if args.save:
            save_dir = evaluation_root / "qualitative" / args.test_dataset.stem
            save_dir.mkdir(exist_ok=True, parents=True)
            save_images([original[0], noisy[0], denoised[0]], save_dir, i)

        psnr, ssim = psnr_ssim_evaluator.psnr_and_ssim(denoised, original)

        metrics['psnr'].append(float(psnr.cpu().numpy()))
        metrics['ssim'].append(float(ssim.cpu().numpy()))

    metrics = {k: statistics.mean(v) for k, v in metrics.items()}

    evaluation_file = evaluation_root / f'denoising_{args.test_dataset.stem}.json'
    with evaluation_file.open('w') as f:
        json.dump(metrics, f, indent='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes a trained denoising model and an evaluation dataset and produces denoising eval results")
    parser.add_argument("model_checkpoint", help="Path to trained model that is to be evaluated")
    parser.add_argument("test_dataset", help="path to json holding pairs of noisy and clean image paths")
    parser.add_argument("dataset_name", help="name of evaluation dataset (e.g. BSD68 or Set12)")
    parser.add_argument("--device", default='cuda', help="device to use")
    parser.add_argument("--save", action='store_true', default=False, help="save reconstructed images together with real images for visual inspection")

    evaluate_denoising(parser.parse_args())
