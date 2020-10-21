from argparse import ArgumentParser


def add_default_args_for_projecting(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr-rampup', type=float, default=0.0)
    parser.add_argument('--lr-rampdown', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--noise-lr', type=float, default=5)
    parser.add_argument('--noise-lr-rampup', type=float, default=0.0)
    parser.add_argument('--noise-lr-rampdown', type=float, default=0.0)
    parser.add_argument('--latent-step', type=int, default=5000)
    parser.add_argument('--noise-step', type=int, default=3000)
    parser.add_argument('--mse', type=float, default=1)
    parser.add_argument('--no-w-plus', dest='w_plus', action='store_false', default=True)
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='batch size for projecting')
    parser.add_argument('--no-mean-latent', action='store_true',
                        help="use pure random latent for start")
    parser.add_argument('--device', default='cuda', help="which device to use")
    parser.add_argument('--config',
                        help='path to a config file if the ckpt is not saved in a location that also contains config info for the train run')
    parser.add_argument('--debug-step', type=int, default=50,
                        help='number of iterations after which to save a debug image')

    return parser
