import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take CBSD68 dir and create multiple evaluation dataset")
    parser.add_argument("cbs_path", help="path to root dir of cbs68 dataset")

    args = parser.parse_args()

    cbs_root = Path(args.cbs_path)
    original_image_dir = cbs_root / 'original_png'
    noisy_image_dirs = cbs_root.glob('noisy*')

    original_pngs = list(sorted(original_image_dir.glob("*.png"), key=lambda x: int(x.stem)))
    for noisy_image_dir in noisy_image_dirs:
        if not noisy_image_dir.is_dir():
            continue

        noisy_pngs = list(sorted(noisy_image_dir.glob("*.png"), key=lambda x: int(x.stem)))

        assert len(original_pngs) == len(noisy_pngs), f"number of original and noisy images is not equal!!, {len(original_pngs)} vs {len(noisy_pngs)}"

        gt = [
            {'original': str(original_png), 'noisy': str(noisy_png)}
            for original_png, noisy_png in zip(original_pngs, noisy_pngs)
        ]

        gt_file_name = cbs_root / f"{noisy_image_dir.parts[-1]}.json"

        with gt_file_name.open('w') as f:
            json.dump(gt, f, indent='\t')
