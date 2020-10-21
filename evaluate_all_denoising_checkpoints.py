import argparse
from pathlib import Path

from tqdm import tqdm

from evaluate_denoising import evaluate_denoising
from utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all denoising checkpoints in given project dir")
    parser.add_argument("project_dir", help="path to project dir")
    parser.add_argument("test_dataset_dir", help="path to test dataset dir with all test splits to run")
    parser.add_argument("dataset_name", help="name of evaluation dataset (e.g. BSD68 or Set12)")

    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    all_checkpoints = list(project_dir.glob("**/*/checkpoints/100000.pt"))

    test_dataset_dir = Path(args.test_dataset_dir)
    test_datasets = list(test_dataset_dir.glob("*.json"))

    evaluate_args = argparse.Namespace()
    evaluate_args.device = 'cuda'
    evaluate_args.save = True
    evaluate_args.dataset_name = args.dataset_name

    for dataset in tqdm(test_datasets):
        evaluate_args.test_dataset = dataset

        for checkpoint in tqdm(all_checkpoints, leave=False):
            config = load_config(checkpoint, None)

            if not (config.get('denoising', False) or config.get('black_and_white_denoising', False)):
                # no denoising checkpoint
                continue

            evaluate_args.model_checkpoint = checkpoint
            evaluate_denoising(evaluate_args)
