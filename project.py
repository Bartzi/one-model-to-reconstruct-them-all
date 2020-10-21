import argparse
import os
from pathlib import Path

import matplotlib
import torch
from PIL import Image
from tqdm import trange

from latent_projecting import run_image_reconstruction, Latents
from latent_projecting.projector import Projector
from utils.command_line_args import add_default_args_for_projecting
from pytorch_training.images.utils import make_image

matplotlib.use('AGG')

Image.init()


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


# def abort_condition(loss_dict):
#     if loss_dict['psnr'] > 10:
#         return True
#     return False


def main(args):
    projector = Projector(args)

    transform = projector.get_transforms()

    imgs = []
    image_names = []

    for file_name in os.listdir(args.files):
        if os.path.splitext(file_name)[-1] not in Image.EXTENSION.keys():
            continue

        image_name = os.path.join(args.files, file_name)
        img = transform(Image.open(image_name).convert('RGB'))
        image_names.append(image_name)
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(args.device)

    n_mean_latent = 10000
    latent_mean, latent_std = projector.get_mean_latent(n_mean_latent)

    for idx in trange(0, len(imgs), args.batch_size):
        images = imgs[idx:idx + args.batch_size]

        base_noises = projector.generator.make_noise()
        base_noises = [noise.repeat(len(images), 1, 1, 1) for noise in base_noises]

        noises = [noise.detach().clone() for noise in base_noises]

        if args.no_mean_latent:
            latent_in = torch.normal(0, latent_std.item(), size=(len(images), projector.config['latent_size']), device=args.device)
        else:
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(len(images), 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, projector.generator.n_latent, 1)

        # optimize latent vector
        paths, best_latent = run_image_reconstruction(args, projector, Latents(latent_in, noises), images, do_optimize_noise=args.optimize_noise)

        # result_file = {'noises': noises}

        img_gen, _ = projector.generator([best_latent.latent.cuda()], input_is_latent=True, noise=[noise.cuda() for noise in best_latent.noise])

        img_ar = make_image(img_gen)

        destination_dir = Path(args.files) / 'projected' / args.destination
        destination_dir.mkdir(parents=True, exist_ok=True)

        path_per_image = paths.split()
        for i in range(len(images)):
            image_name = image_names[idx + i]
            image_latent = best_latent[i]
            result_file = {
                'noise': image_latent.noise,
                'latent': image_latent.latent,
            }
            image_base_name = os.path.splitext(os.path.basename(image_name))[0]
            img_name = image_base_name + '-project.png'
            pil_img = Image.fromarray(img_ar[i])
            pil_img.save(destination_dir / img_name)
            torch.save(result_file, destination_dir / f'results_{image_base_name}.pth')
            if args.create_gif:
                projector.create_gif(
                    path_per_image[i].to(args.device),
                    image_base_name,
                    destination_dir
                )
            projector.render_log(destination_dir, image_base_name)

        # cleanup
        del paths
        del best_latent
        torch.cuda.empty_cache()
        projector.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILES', help="Path to dir holding all images to embed")
    parser.add_argument('destination', help="name of the destination subdir where results will be saved")
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--create-gif', help='create gif showing the optimization process', action='store_true', default=False)
    parser.add_argument('--no-noise-optimize', action='store_false', default=True, dest='optimize_noise', help="do not perform noise optimization")

    parser = add_default_args_for_projecting(parser)

    args = parser.parse_args()
    assert not Path(args.destination).is_absolute(), "The destination path is supposed to be a relative path!"
    main(args)
