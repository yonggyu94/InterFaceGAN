# python3.7
"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.

NOTE: If you want to use W or W+ space of StyleGAN, please do not randomly
sample the latent code, since neither W nor W+ space is subject to Gaussian
distribution. Instead, please use `generate_data.py` to get the latent vectors
from W or W+ space first, and then use `--input_latent_codes_path` option to
pass in the latent vectors.
"""

import os.path
import argparse
from tqdm import tqdm

from utils.logger import setup_logger
from utils.manipulator import linear_interpolate


# ------------------------------------------------------
import os

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
# -------------------------------------------------------


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        # choices=list(MODEL_POOL),
                        help='Name of the model for generation. (required)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-b', '--boundary_path', type=str, required=True,
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                        help='If specified, will load latent codes from given '
                             'path instead of randomly sampling. (optional)')
    parser.add_argument('-n', '--num', type=int, default=1,
                        help='Number of images for editing. This field will be '
                             'ignored if `input_latent_codes_path` is specified. '
                             '(default: 1)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('--start_distance', type=float, default=-3.0,
                        help='Start point for manipulation in latent space. '
                             '(default: -3.0)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for image editing. (default: 10)')

    # ----------------------------------------------------------------------------

    parser.add_argument('--network_pkl', type=str, default="network_128_025000.pkl",
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_args()


def preprocess(latent_codes, latent_space_type='Z'):
    """Preprocesses the input latent code if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

    Returns:
      The preprocessed latent codes which can be used as final input for the
        generator.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    if not isinstance(latent_codes, np.ndarray):
        raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_space_dim = 512
    w_space_dim = 512
    num_layers = 12         # 12, 14, 16...

    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
        latent_codes = latent_codes.reshape(-1, latent_space_dim)
        norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
        latent_codes = latent_codes / norm * np.sqrt(latent_space_dim)
    elif latent_space_type == 'W':
        latent_codes = latent_codes.reshape(-1, w_space_dim)
    elif latent_space_type == 'WP':
        latent_codes = latent_codes.reshape(-1, num_layers, w_space_dim)
    else:
        raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)


def get_batch_inputs(latent_codes, batch_size):
    """Gets batch inputs from a collection of latent codes.

    This function will yield at most `self.batch_size` latent_codes at a time.

    Args:
      latent_codes: The input latent codes for generation. First dimension
        should be the total number.
    """
    total_num = latent_codes.shape[0]
    for i in range(0, total_num, batch_size):
        yield latent_codes[i:i + batch_size]


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='generate_data')

    logger.info(f'Initializing generator.')
    # gan_type = MODEL_POOL[args.model_name]['gan_type']
    gan_type = args.model_name
    if gan_type == 'stylegan2':
        print('Loading networks from "%s"...' % args.network_pkl)
        device = torch.device('cuda')
        with dnnlib.util.open_url(args.network_pkl) as f:
            model = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    else:
        raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

    logger.info(f'Preparing boundary.')
    if not os.path.isfile(args.boundary_path):
        raise ValueError(f'Boundary `{args.boundary_path}` does not exist!')
    boundary = np.load(args.boundary_path)
    np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)

    logger.info(f'Preparing latent codes.')
    if os.path.isfile(args.input_latent_codes_path):
        logger.info(f'  Load latent codes from `{args.input_latent_codes_path}`.')
        latent_codes = np.load(args.input_latent_codes_path)
        # latent_codes = model.preprocess(latent_codes, **kwargs)   # StyleGAN1, ProgressiveGAN
        latent_codes = preprocess(latent_codes, latent_space_type='W')  # StyleGAN2
    else:
        logger.info(f'  Sample latent codes randomly.')
        latent_codes = model.easy_sample(args.num, **kwargs)

    np.save(os.path.join(args.output_dir, 'latent_codes.npy'), latent_codes)
    total_num = latent_codes.shape[0]

    logger.info(f'Editing {total_num} samples.')
    for sample_id in tqdm(range(total_num), leave=False):
        interpolations = linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                            boundary,
                                            start_distance=args.start_distance,
                                            end_distance=args.end_distance,
                                            steps=args.steps)
        interpolation_id = 0
        for interpolations_batch in get_batch_inputs(interpolations, args.batch_size):
            if gan_type == 'stylegan2':
                interpolations_batch = torch.from_numpy(interpolations_batch)
                interpolations_batch = interpolations_batch.unsqueeze(1).repeat(1, 12, 1).to(device)
                img = model.synthesis(interpolations_batch)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            for idx in range(len(interpolations_batch)):
                save_path = os.path.join(args.output_dir,
                                         f'{sample_id:03d}_{interpolation_id:03d}.jpg')

                PIL.Image.fromarray(img[idx].cpu().numpy(), 'RGB').save(save_path)
                interpolation_id += 1

        assert interpolation_id == args.steps
        logger.debug(f'  Finished sample {sample_id:3d}.')
    logger.info(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
    main()
