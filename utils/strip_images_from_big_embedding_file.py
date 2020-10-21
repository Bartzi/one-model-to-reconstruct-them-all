import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take a file with many image embeddings and extract a given number of image embeddings from it")
    parser.add_argument("embedding_file", help="path to embedding file")
    parser.add_argument("-n", "--num-samples", type=int, default=100, help="number of embeddings to extract")

    args = parser.parse_args()

    embedded_data = np.load(args.embedding_file, mmap_mode='r')

    image_data = {key: embedded_data[key][:args.num_samples] for key in tqdm(list(embedded_data.keys()))}

    embedding_path = Path(args.embedding_file)
    embedding_name_parts = embedding_path.stem.split('_')
    embedding_name_parts[0] = 'small_embedding'
    new_embedding_name = '_'.join(embedding_name_parts)
    with (embedding_path.parent / f"{new_embedding_name}.npz").open('wb') as f:
        np.savez(f, **image_data)
