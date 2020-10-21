import argparse
import json
import random
from pathlib import Path
from typing import Tuple, List

import matplotlib
from scipy import stats
from scipy.interpolate import interpolate

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import torch
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm, trange

from latent_projecting import Latents
from networks import load_weights, get_autoencoder
from utils.config import load_config
from utils.data_loading import build_data_loader


def embed_images(args: argparse.Namespace, config: dict, dataset: Path) -> Tuple[numpy.ndarray, List[numpy.ndarray], numpy.ndarray]:
    data_loader = build_data_loader(dataset, config, config['absolute'], shuffle_off=True)
    if args.num_samples is not None:
        random.seed(args.seed if args.seed != 'none' else None)
        data_loader.dataset.image_data = random.sample(data_loader.dataset.image_data, args.num_samples)

    autoencoder = get_autoencoder(config).to(args.device)
    autoencoder = load_weights(autoencoder, args.autoencoder_checkpoint, key='autoencoder')

    latent_codes = []
    noises = None
    image_names = []

    for idx, batch in enumerate(tqdm(data_loader)):
        if isinstance(batch, dict):
            batch = batch['input_image']
        batch = batch.to(args.device)
        with torch.no_grad():
            latents: Latents = autoencoder.encode(batch)
        latents = latents.numpy()
        latent_codes.append(latents.latent)
        if noises is None:
            noises = [[noise] for noise in latents.noise]
        else:
            for noise, latent_noise in zip(noises, latents.noise):
                noise.append(latent_noise)

        for batch_idx in range(len(batch)):
            image_names.append(data_loader.dataset.image_data[idx * config['batch_size'] + batch_idx])

    latent_codes = numpy.concatenate(latent_codes, axis=0)
    noises = [numpy.concatenate(noise, axis=0) for noise in noises]

    return latent_codes, noises, numpy.array(image_names)


def create_cdf(data: numpy.ndarray) -> ECDF:
    return ECDF(data.ravel())


def create_and_plot_cdf(data: numpy.ndarray, file_name: Path):
    cdf = create_cdf(data)
    plt.clf()
    plt.plot(cdf.x, cdf.y)
    plt.savefig(file_name)


def create_and_save_histogram(data: numpy.ndarray, file_name: Path, add_inverse_cdf_results: bool = False):
    plt.clf()
    n, bins, patches = plt.hist(data.ravel(), bins='auto')
    if add_inverse_cdf_results:
        inverse_cdf = get_inverse_cdf(data.ravel())
        approximated_data_dist = inverse_cdf(numpy.random.rand(data.size))
        plt.hist(approximated_data_dist.ravel(), bins=bins)

    plt.savefig(file_name)


def create_inverse_transform_building_blocks(data: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    hist, bin_edges = numpy.histogram(data, bins='auto', density=True)
    cum_values = numpy.zeros(bin_edges.shape)
    cum_values[1:] = numpy.cumsum(hist * numpy.diff(bin_edges))
    return cum_values, bin_edges


def get_inverse_cdf(data: numpy.ndarray) -> interpolate.interp1d:
    cum_values, bin_edges = create_inverse_transform_building_blocks(data)
    inverse_cdf = interpolate.interp1d(cum_values, bin_edges)
    return inverse_cdf


def save_array_as_image(data: numpy.ndarray, path: Path, overwrite: bool = False, **kwargs):
    if path.exists() and not overwrite:
        return
    plt.clf()
    plt.figure(figsize=(20, 16))
    plt.matshow(data)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.savefig(path, dpi=300)


def normalize_data(data: numpy.ndarray, eps: float = 1e-9, **kwargs) -> numpy.ndarray:
    max_values = numpy.max(data, **kwargs)
    min_values = numpy.min(data, **kwargs)
    return (data - min_values) / (max_values - min_values + eps)


def parallel_coordinate_plot(data: numpy.ndarray, path: Path, axis: int = 0, overwrite: bool = False,
                             normalize: bool = False, **kwargs):
    if path.exists() and not overwrite:
        return
    if normalize:
        data = normalize_data(data, axis=axis)
    plt.clf()
    slicing = [slice(None, None, None) for _ in range(data.ndim - 1)]
    for n in range(data.shape[axis]):
        data_row_shape = tuple(slicing[:axis] + [n] + slicing[axis:])
        plt.plot(data[data_row_shape], **kwargs)
    plt.savefig(path)


class Analyzer:

    def __init__(self, dest_dir: Path, save_suffix: str, check_reconstructed_cdf: bool = False,
                 disable_histograms: bool = False, disable_blueprints: bool = False, disable_stats: bool = False,
                 force: bool = False, force_tests: bool = False):
        self.dest_dir = dest_dir
        self.save_suffix = save_suffix

        self.check_reconstructed_cdf = check_reconstructed_cdf
        self.disable_histograms = disable_histograms
        self.disable_blueprints = disable_blueprints
        self.disable_stats = disable_stats
        self.force = force
        self.force_tests = force_tests


class LatentCoderAnalyzer(Analyzer):

    def __init__(self, *args, w_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.histogram_dir = self.dest_dir / 'latent_histograms'
        self.histogram_dir.mkdir(exist_ok=True, parents=True)

        self.w_only = w_only

    def create_histogram_per_latent_dim(self, array: numpy.ndarray, root_dir: Path):
        assert len(array.shape) == 2, "In order to create histogram per dim only two dimensions are allowed (num samples, latent_size)"
        histogram_dir = root_dir / 'histograms'
        histogram_dir.mkdir(parents=True, exist_ok=True)
        cdf_dir = root_dir / 'cdf'
        cdf_dir.mkdir(parents=True, exist_ok=True)

        for i in trange(array.shape[-1], leave=False):
            file_name = f'{i:06}.png'
            histogram_name = histogram_dir / file_name
            if not histogram_name.exists() or self.force:
                create_and_save_histogram(array[:, i], histogram_name, add_inverse_cdf_results=self.check_reconstructed_cdf)

            cdf_name = cdf_dir / file_name
            if not cdf_name.exists() or self.force:
                create_and_plot_cdf(array[:, i], cdf_name)

    def create_latent_histograms(self, latent_codes: numpy.ndarray):
        full_latent_path = self.histogram_dir / '0000000_full_latent_codes.png'
        if not full_latent_path.exists() or self.force:
            create_and_save_histogram(latent_codes, full_latent_path)

        if self.w_only:
            latent_codes = latent_codes[:, numpy.newaxis, ...]

        for i in trange(latent_codes.shape[1]):
            name = self.histogram_dir / f'000000_w_plus_latent_slice_{i}.png'
            if not name.exists() or self.force:
                create_and_save_histogram(latent_codes[:, i, ...], name)

            per_latent_dim_dir = self.histogram_dir / "analysis_per_dim" / str(i)
            per_latent_dim_dir.mkdir(parents=True, exist_ok=True)
            self.create_histogram_per_latent_dim(latent_codes[:, i, ...], per_latent_dim_dir)

    def create_latent_code_blueprint(self, latent_codes: numpy.ndarray) -> dict:
        blueprint = {
            'w_only': self.w_only,
            'type': 'latent',
        }
        if self.w_only:
            latent_codes = latent_codes[:, numpy.newaxis, ...]

        per_dim = {}
        for w_slice in trange(latent_codes.shape[1]):
            for dimension in trange(latent_codes.shape[2], leave=False):
                key = str(f"{w_slice}_{dimension}")
                array = latent_codes[:, w_slice, dimension]
                std = array.std()
                if std < 1e-7:
                    per_dim[key] = {
                        "value": float(array.mean())
                    }
                else:
                    cum_values, bin_edges = create_inverse_transform_building_blocks(array)
                    per_dim[key] = {
                        "cum_values": cum_values.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    }

        blueprint['blueprint'] = per_dim
        return blueprint

    def __call__(self, latent_codes: numpy.ndarray):
        if not self.disable_histograms:
            print("creating latent histograms")
            self.create_latent_histograms(latent_codes)

        if not self.disable_stats:
            if self.w_only:
                self.plots_and_stats(latent_codes[:, numpy.newaxis, :], 'latent_code')
            else:
                self.plots_and_stats(latent_codes, 'latent_code')

        if self.disable_blueprints:
            return
        print("building latent code blueprint")
        blueprint = self.create_latent_code_blueprint(latent_codes)

        blueprint_file_name = self.dest_dir / f'latent_blueprint_{self.save_suffix}.json'
        with blueprint_file_name.open('w') as f:
            json.dump(blueprint, f)

    def plots_and_stats(self, data: numpy.ndarray, parent_dir: str, max_samples_processed: int = 1000,
                        max_variables_processed: int = 128, variables_per_plot: int = 16):
        plot_dir = self.dest_dir / "visualizations" / parent_dir
        plot_dir.mkdir(exist_ok=True, parents=True)

        results_dir = self.dest_dir / "test_results" / parent_dir
        results_dir.mkdir(exist_ok=True, parents=True)

        num_samples, slices, code_length = data.shape

        normalized_data = normalize_data(data, axis=0)

        print("parallel coordinate plots")
        for k in tqdm(range(0, max_variables_processed, variables_per_plot)):
            for i in range(slices):
                parallel_coordinate_plot(
                    normalized_data[0:max_samples_processed, i, k:(k + variables_per_plot)],
                    plot_dir / f"plot_a_{k}-{k + variables_per_plot}_{i}.png",
                    axis=0, c="k", alpha=0.1, lw=0.1
                )

        if slices > 1:
            print("parallel coordinate plots (per slice)")
            for j in tqdm(range(max_variables_processed)):
                parallel_coordinate_plot(
                    normalized_data[0:max_samples_processed, :, j],
                    plot_dir / f"plot_b_{j}.png",
                    axis=0, c="k", alpha=0.1, lw=0.1
                )

        print("spearman correlation over latent code")
        for i in tqdm(range(slices)):
            correlation_result, p_value = stats.spearmanr(data[0:max_samples_processed, i, :], axis=0)
            save_array_as_image(correlation_result, results_dir / f'correlation_c_{i}.png', overwrite=self.force_tests)

        print("test for normal distribution")
        results_all = numpy.zeros([slices, 1])
        results_per_code = numpy.zeros([slices, code_length])
        results_per_sample = numpy.zeros([slices, max_samples_processed])
        for i in tqdm(range(slices)):
            w, p_value = stats.shapiro(data[:, i, :])
            results_all[i, 0] = p_value

            for j in range(code_length):
                w, p_value = stats.shapiro(data[:, i, j])
                results_per_code[i, j] = p_value

            for j in range(max_samples_processed):
                w, p_value = stats.shapiro(data[j, i, :])
                results_per_sample[i, j] = p_value

        save_array_as_image(results_all, results_dir / 'shapiro_wilk_sliced.png', overwrite=self.force_tests)
        save_array_as_image(results_per_code, results_dir / 'shapiro_wilk_per_code.png', overwrite=self.force_tests)
        save_array_as_image(results_per_sample, results_dir / 'shapiro_wilk_per_sample.png', overwrite=self.force_tests)


class NoiseAnalyzer(Analyzer):

    def __init__(self, noise_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_key = noise_key

        self.histogram_dir = self.dest_dir / 'noise_histograms' / noise_key
        self.histogram_dir.mkdir(parents=True, exist_ok=True)

    def create_noise_histograms(self, noises: numpy.ndarray):
        full_noise_path = self.histogram_dir / "000000_full_noise.png"
        if not full_noise_path.exists() or self.force:
            create_and_save_histogram(noises, full_noise_path)

        num_histograms = noises.shape[-2] * noises.shape[-1]
        dir_per_row = num_histograms > 10000

        per_pixel_dir = self.histogram_dir / "histograms_per_pixel"
        per_pixel_dir.mkdir(parents=True, exist_ok=True)
        for y in trange(noises.shape[-2]):
            if dir_per_row:
                per_pixel_dir = self.histogram_dir / "histograms_per_pixel" / str(y)
                per_pixel_dir.mkdir(parents=True, exist_ok=True)
            for x in trange(noises.shape[-1], leave=False):
                dest_name = per_pixel_dir / f"{x}_{y}.png"
                create_and_save_histogram(noises[:, :, y, x], dest_name, add_inverse_cdf_results=self.check_reconstructed_cdf)

    def create_noise_blueprint(self, noises: numpy.ndarray) -> dict:
        blueprint = {
            'type': 'noise',
            'key': self.noise_key,
        }

        per_pixel = {}
        for y in trange(noises.shape[-2]):
            for x in trange(noises.shape[-1], leave=False):
                key = f"{x}_{y}"
                array = noises[:, :, y, x]
                array = numpy.ascontiguousarray(array)

                if array.std() < 1e-8:
                    per_pixel[key] = {
                        'value': float(array.mean())
                    }
                else:
                    cum_values, bin_edges = create_inverse_transform_building_blocks(array)
                    per_pixel[key] = {
                        'cum_values': cum_values.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }

        blueprint['blueprint'] = per_pixel
        return blueprint

    def __call__(self, noises: numpy.ndarray):
        if not self.disable_histograms:
            print(f"creating noise histograms for {self.noise_key}")
            self.create_noise_histograms(noises)

        if not self.disable_stats:
            self.plots_and_stats(noises.squeeze(), self.noise_key)

        if self.disable_blueprints:
            return
        print(f"creating noise blueprint for {self.noise_key}")
        blueprint = self.create_noise_blueprint(noises)
        blueprint_file_name = self.dest_dir / f'noise_blueprint_{self.save_suffix}_{self.noise_key}.json'
        with blueprint_file_name.open('w') as f:
            json.dump(blueprint, f)

    def plots_and_stats(self, data: numpy.ndarray, parent_dir: str, max_samples_processed: int = 1000,
                        max_dim_processed: int = 32, max_len_per_plot: int = 16):
        plot_dir = self.dest_dir / "visualizations" / parent_dir
        plot_dir.mkdir(exist_ok=True, parents=True)

        results_dir = self.dest_dir / "test_results" / parent_dir
        results_dir.mkdir(exist_ok=True, parents=True)

        num_samples, _, dim_size = data.shape
        dim_size_processed = min(dim_size, max_dim_processed)

        normalized_data = normalize_data(data, axis=0)

        print("parallel coordinate plots")
        for k in tqdm(range(0, dim_size_processed, max_len_per_plot)):
            for i in range(dim_size_processed):
                parallel_coordinate_plot(
                    normalized_data[0:max_samples_processed, i, k:(k + max_len_per_plot)],
                    plot_dir / f"plot_x_{k}-{k + max_len_per_plot}_{i}.png",
                    axis=0, c="k", alpha=0.1, lw=0.1
                )
                parallel_coordinate_plot(
                    normalized_data[0:max_samples_processed, k:(k + max_len_per_plot), i],
                    plot_dir / f"plot_y_{k}-{k + max_len_per_plot}_{i}.png",
                    axis=0, c="k", alpha=0.1, lw=0.1
                )

        print("spearman correlation over noise")
        for i in tqdm(range(dim_size_processed)):
            correlation_result, p_value = stats.spearmanr(data[0:max_samples_processed, i, :], axis=0)
            save_array_as_image(correlation_result, results_dir / f'correlation_x_{i}.png', overwrite=self.force_tests)
            correlation_result, p_value = stats.spearmanr(data[0:max_samples_processed, :, i], axis=0)
            save_array_as_image(correlation_result, results_dir / f'correlation_y_{i}.png', overwrite=self.force_tests)


def main(args):
    dest_dir = Path(args.autoencoder_checkpoint).parent.parent / args.save_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.autoencoder_checkpoint, None)
    dataset = Path(args.dataset if args.dataset is not None else config['images'])

    save_suffix = f"{Path(args.autoencoder_checkpoint).stem}_{dataset.stem}_{args.num_samples}"

    embedding_file_name = dest_dir / f'embedding_{save_suffix}.npz'

    if not embedding_file_name.exists() or args.force:
        print("finding codes")
        latent_codes, noises, image_names = embed_images(args, config, dataset)
        with embedding_file_name.open('wb') as f:
            noise_files = {}
            for noise in noises:
                key = f"noise_{(shape := noise.shape)[-2]}_{shape[-1]}"
                if key in noise_files:
                    noise_files[key] = numpy.concatenate([noise_files[key], noise], axis=1)
                else:
                    noise_files[key] = noise
            numpy.savez_compressed(f, latent_codes=latent_codes, image_names=image_names, **noise_files)
        del latent_codes
        del noises
        del image_names

    print("loading from saved file")
    data = numpy.load(embedding_file_name, mmap_mode='r')
    latent_codes = data['latent_codes']

    latent_analyzer = LatentCoderAnalyzer(
        dest_dir,
        save_suffix,
        w_only=config['w_only'],
        force=args.force_latent,
        check_reconstructed_cdf=args.check,
        disable_histograms=args.disable_histograms,
        disable_blueprints=args.disable_blueprints,
        disable_stats=args.disable_stats,
        force_tests=args.force_test_results,
    )
    latent_analyzer(latent_codes)
    del latent_codes

    for key in data.keys():
        if 'noise_' in key:
            noise_data = data[key]
            for noise_id in range(noise_data.shape[1]):
                noise_analyzer = NoiseAnalyzer(
                    f"{key}_{noise_id}",
                    dest_dir,
                    save_suffix,
                    force=args.force_noise,
                    check_reconstructed_cdf=args.check,
                    disable_histograms=args.disable_histograms,
                    disable_blueprints=args.disable_blueprints,
                    disable_stats=args.disable_stats,
                )
                noise = noise_data[:, noise_id, ...]
                if len(noise.shape) == 3:
                    noise = noise[:, numpy.newaxis, ...]
                noise_analyzer(noise)
            del noise_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool that uses autoencoder to create latent code for a given dataset, produces histograms of the latent codes and finds the distribution for sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("autoencoder_checkpoint", help='Path to autoencoder checkpoint that is to be analyzed')
    parser.add_argument("--dataset", help="path to dataset to extract latent code of (use only if you do not want to use train dataset")
    parser.add_argument("--save-dir", default='latent_code_analysis', help='name of dir, where resulting histograms and latent codes shall be saved')
    parser.add_argument("-d", "--device", default='cuda', help='Use CPU or GPU for embedding')
    parser.add_argument("-f", "--force", action='store_true', default=False, help="force rerun of embedding with model")
    parser.add_argument("-fl", "--force-latent", action='store_true', default=False, help="force recreation of latent histograms")
    parser.add_argument("-fn", "--force-noise", action='store_true', default=False, help="force recreation of noise histograms")
    parser.add_argument("-ft", "--force-test-results", action='store_true', default=False, help="force recreation of test results")
    parser.add_argument("--disable-all", action='store_true', default=False, help="only create embeddings")
    parser.add_argument("--disable-histograms", action='store_true', default=False, help="do not create any histograms")
    parser.add_argument("--disable-blueprints", action='store_true', default=False, help="do not create blueprints")
    parser.add_argument("--disable-stats", action='store_true', default=False, help="do not create plots and test for statistics")
    parser.add_argument("-n", "--num-samples", type=int, help="restrict the number of samples to evaluate to this number")
    parser.add_argument("--seed", default="100", help="seed to use for random sampling if only evaluating on subset of data (set to 'none' to turn off)")
    parser.add_argument("--check", action='store_true', default=False, help="check whether inverse transform sampling is able to correctly approximate the true distribution")

    args = parser.parse_args()
    args.disable_histograms |= args.disable_all
    args.disable_blueprints |= args.disable_all
    args.disable_stats |= args.disable_all
    main(args)
