import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finds all trained checkpoints in given project log dir")
    parser.add_argument("project_dir", help="path to project dir that is to be analyzed")

    args = parser.parse_args()
    project_dir = Path(args.project_dir)

    runs = [d for d in project_dir.iterdir() if d.is_dir()]
    all_checkpoints = []
    for run in runs:
        checkpoints = list(run.glob('**/checkpoints/*.pt'))
        all_checkpoints.extend(checkpoints)

    with (project_dir / "trained_checkpoints.txt").open('w') as f:
        for checkpoint in all_checkpoints:
            print(checkpoint.resolve(), file=f)
