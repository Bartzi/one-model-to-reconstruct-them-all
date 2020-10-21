import argparse

import numpy as np
from scipy import stats
from tqdm import tqdm


def main(args):
    with np.load(args.latent_codes, mmap_mode='r') as data:
        # available arrays:
        #   latent_codes, image_names,
        #   noise_4_4, noise_8_8, noise_16_16, noise_32_32, noise_64_64, noise_128_128, noise_256_256
        latent_codes = data["latent_codes"]
        num_samples, slices, code_length = latent_codes.shape
        print("shape:", num_samples, slices, code_length)
        print("normal distribution can be assumed if second value is larger than 0.05")
        for i in range(slices):
            shapiro_test = stats.shapiro(latent_codes[:,i,:])
            print("samples :", "slice:", i, "latent_space :", "result:", shapiro_test)
        for i in range(slices):
            shapiro_test = stats.shapiro(latent_codes[:, i, 130])
            print("samples :", "slice:", i, "latent_space 0", "result:", shapiro_test)
        for i in range(slices):
            shapiro_test = stats.shapiro(latent_codes[0, i, :])
            print("samples 0", "slice:", i, "latent_space :", "result:", shapiro_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test whether the latent codes are a normal distribution with shapiro wilk test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("latent_codes", help='Path to file which contains latent codes')
    parser.add_argument("-d", "--device", default='cuda', help='Use CPU or GPU for embedding')

    main(parser.parse_args())
