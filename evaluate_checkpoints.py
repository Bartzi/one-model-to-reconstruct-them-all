import argparse
import copy
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from tqdm.contrib import tenumerate

from data import get_dataset_class
from evaluation.fid import FID
from evaluation.psnr_ssim import PSNRSSIMEvaluator
from networks import get_autoencoder, load_weights, StyleganAutoencoder
from utils.config import load_config
from utils.data_loading import build_data_loader


def save_eval_result(eval_result: dict, eval_type: str, dest_dir: Path, dataset_name: str, checkpoint_name: str):
    dest_file = dest_dir / f"{eval_type}.json"
    if dest_file.exists():
        with dest_file.open("r") as f:
            json_data = json.load(f)
    else:
        json_data = {}

    checkpoint_results = json_data.get(checkpoint_name, {})
    checkpoint_results[dataset_name] = eval_result
    json_data[checkpoint_name] = checkpoint_results

    with dest_file.open("w") as f:
        json.dump(json_data, f, indent='\t')


def evaluate_reconstruction(autoencoder: StyleganAutoencoder, data_loaders: dict) -> dict:
    metrics = defaultdict(list)
    psnr_ssim_evaluator = PSNRSSIMEvaluator()

    for i, batch in tenumerate(data_loaders['test'], desc="psnr_ssim", leave=False):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            denoised = autoencoder(batch['input_image'])

        psnr, ssim = psnr_ssim_evaluator.psnr_and_ssim(denoised, batch['output_image'])

        metrics['psnr'].append(float(psnr.cpu().numpy()))
        metrics['ssim'].append(float(ssim.cpu().numpy()))
    metrics = {k: statistics.mean(v) for k, v in metrics.items()}
    return metrics


def evaluate_fid(autoencoder: StyleganAutoencoder, data_loaders: dict, dataset: dict) -> dict:
    fid_evaluator = FID(num_samples=50_000)

    test_dataset_len = len(data_loaders['test'])
    if test_dataset_len > 50_000:
        data_key = 'test'
    else:
        if len(data_loaders['train']) < 50_000:
            print("warning test and train dataset are smaller than 50.000 samples!")
        data_key = 'train'

    fid_scores = fid_evaluator(autoencoder, data_loaders[data_key], dataset[data_key])
    return {"fid": fid_scores}


def has_not_been_evaluated(checkpoint_name: str, dataset_name: str, evaluation_root: Path) -> Dict[str, bool]:
    already_done_map = {}
    for eval_type in ["fid", "reconstruction"]:
        dest_file = evaluation_root / f"{eval_type}.json"
        if not dest_file.exists():
            already_done_map[eval_type] = True
            continue

        with dest_file.open() as f:
            evaluation_data = json.load(f)

        if checkpoint_name not in evaluation_data:
            already_done_map[eval_type] = True
            continue

        evaluation_data = evaluation_data[checkpoint_name]
        already_done_map[eval_type] = dataset_name not in evaluation_data

    return already_done_map


def evaluate_checkpoint(checkpoint: str, dataset: dict, args: argparse.Namespace):
    checkpoint = Path(checkpoint)
    train_run_root_dir = checkpoint.parent.parent
    evaluation_root = train_run_root_dir / 'evaluation'
    evaluation_root.mkdir(exist_ok=True)

    dataset_name = dataset.pop('name')
    to_evaluate = has_not_been_evaluated(checkpoint.name, dataset_name, evaluation_root)
    if not args.fid:
        to_evaluate['fid'] = False
    if not args.reconstruction:
        to_evaluate['reconstruction'] = False

    if not any(to_evaluate.values()):
        # there is nothing to evaluate
        return

    config = load_config(checkpoint, None)

    dataset = {k: Path(v) for k, v in dataset.items()}

    autoencoder = get_autoencoder(config).to('cuda')
    autoencoder = load_weights(autoencoder, checkpoint, key='autoencoder', strict=True)

    config['batch_size'] = 1

    dataset_class = get_dataset_class(argparse.Namespace(**config))
    data_loaders = {
        key: build_data_loader(value, config, config['absolute'], shuffle_off=True, dataset_class=dataset_class)
        for key, value in dataset.items()
    }

    if to_evaluate['fid']:
        fid_result = evaluate_fid(autoencoder, data_loaders, dataset)
        save_eval_result(fid_result, "fid", evaluation_root, dataset_name, checkpoint.name)

    if to_evaluate['reconstruction']:
        reconstruction_result = evaluate_reconstruction(autoencoder, data_loaders)
        save_eval_result(reconstruction_result, "reconstruction", evaluation_root, dataset_name, checkpoint.name)

    del autoencoder
    torch.cuda.empty_cache()


def main(args):
    checkpoint_file = Path(args.checkpoint_list)
    with checkpoint_file.open() as f:
        checkpoints = [line.rstrip() for line in f]

    dataset_file = Path(args.dataset_file)
    with dataset_file.open() as f:
        datasets = json.load(f)

    failed_combinations = []
    try:
        for checkpoint, dataset in tqdm(itertools.product(checkpoints, datasets), total=len(checkpoints) * len(datasets)):
            try:
                evaluate_checkpoint(checkpoint, copy.deepcopy(dataset), args)
            except Exception as e:
                failed_combinations.append({"combination": (checkpoint, dataset), "reason": str(e)})
    finally:
        for combination in failed_combinations:
            print(f"The following eval combination failed: {combination['combination']}, with reason: {combination['reason']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes a list of checkpoints and datasets and runs evaluation")
    parser.add_argument("checkpoint_list", help="path to file that contains the path to a trained checkpoint in each line")
    parser.add_argument("dataset_file", help="path to json file that contains the paths to datasets each model is to be evaluated on")
    parser.add_argument("--skip-fid", dest="fid", action='store_false', default=True, help="skip fid during evaluation")
    parser.add_argument("--skip-reconstruction", dest="reconstruction", action='store_false', default=True, help="skip calculation of reconstruction metrics such as PSNR and SSID")

    main(parser.parse_args())
