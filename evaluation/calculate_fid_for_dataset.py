import argparse
import json
from pathlib import Path

from evaluation.fid import FID
from networks import get_autoencoder, load_weights
from utils.config import load_config
from utils.data_loading import build_data_loader


def save_fid_score(fid_score: float, dest_dir: Path, dataset_name: str):
    dest_file = dest_dir / "fid.json"
    if dest_file.exists():
        with dest_file.open("r") as f:
            json_data = json.load(f)
    else:
        json_data = {}

    if dataset_name in json_data:
        print("WARNING: Already found an FID result for this dataset")
        while True:
            answer = input("Overwrite [y|N]? ")
            if len(answer) == 0 or answer.lower() == 'n':
                return
            elif answer.lower() == 'y':
                break
            print(f"Did not understand: {answer}")
    json_data[dataset_name] = fid_score

    with dest_file.open("w") as f:
        json.dump(json_data, f, indent='\t')


def main(args: argparse.Namespace):
    dest_dir = Path(args.model_checkpoint).parent.parent / 'evaluation'
    dest_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.model_checkpoint, None)
    dataset = Path(args.dataset)

    config['batch_size'] = args.batch_size
    data_loader = build_data_loader(dataset, config, config['absolute'], shuffle_off=True)
    fid_calculator = FID(args.num_samples, device=args.device)

    autoencoder = get_autoencoder(config).to(args.device)
    autoencoder = load_weights(autoencoder, args.model_checkpoint, key='autoencoder')

    fid_score = fid_calculator(autoencoder, data_loader, args.dataset)

    save_fid_score(fid_score, dest_dir, args.dataset_name)

    print(f"FID Score for {args.dataset_name} is {fid_score}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that calculates FID metric for a given model and dataset")
    parser.add_argument("model_checkpoint", help="path to the model that is to be analyzed")
    parser.add_argument("dataset", help="path to json holding dataset information")
    parser.add_argument("dataset_name", help="human readable name of dataset you are evaluating (used for saving the results)")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch Size for forwarding through model")
    parser.add_argument("-n", "--num-samples", type=int, default=1000, help="number of samples to use for FID calculation")
    parser.add_argument("-d", "--device", default='cuda', help="device to use")

    main(parser.parse_args())
