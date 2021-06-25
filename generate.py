# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

# ----------------------------------------------------------------------------


def num_range(s: str) -> List[int]:
    """Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints."""

    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--seeds", type=num_range, help="List of random seeds")
@click.option(
    "--latent",
    "latent",
    type=bool,
    help="whether to perform latent space exploration or random generation",
    default=False,
    show_default=True,
)
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option(
    "--frames",
    "frames",
    type=int,
    help="number of samples per seed",
    default=1,
    show_default=True,
)
@click.option(
    "--class",
    "class_idx",
    type=int,
    help="Class label (unconditional if not specified)",
)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option("--projected-w", help="Projection result file", type=str, metavar="FILE")
@click.option(
    "--outdir",
    help="Where to save the output images",
    type=str,
    required=True,
    metavar="DIR",
)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    latent: bool,
    seeds: Optional[List[int]],
    truncation_psi: float,
    frames: int,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    # device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print("warn: --seeds is ignored when using --projected-w")
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)["w"]
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
                f"{outdir}/proj{idx:02d}.png"
            )
        return

    if seeds is None:
        ctx.fail("--seeds option is required when not using --projected-w")

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail(
                "Must specify class label with --class when using a conditional network"
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    if latent:
        # seed to latent vectors
        zs = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            z = rng.randn(1, G.z_dim)
            zs.append(z)

        # convert to linspace
        num_walk = len(zs) - 1
        walk_vectors = []
        sample = int(frames / num_walk) + 1
        for nw in range(num_walk):
            z0, z1 = zs[nw + 0], zs[nw + 1]
            walk_vector = np.linspace(z0, z1, sample)
            walk_vectors.append(walk_vector)

        # generate images
        for nw in range(num_walk):
            latent_z = walk_vectors[nw].astype("float32")
            seed0, seed1 = seeds[nw + 0], seeds[nw + 1]
            print(
                "Generating images for seed %d-%d (%d/%d)..."
                % (seed0, seed1, nw, num_walk)
            )
            for vector_index in range(latent_z.shape[0]):
                print("Generating image %d..." % (vector_index))
                vector = latent_z[vector_index].reshape((1, -1)).to(device)
                img = G(
                    vector,
                    label,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode,
                    force_fp32=True,
                )
                # converting image to uint8
                img = (
                    (img.permute(0, 2, 3, 1) * 127.5 + 128)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                out_image = img[0].cpu().numpy()
                PIL.Image.fromarray(out_image, "RGB").save(
                    f"{outdir}/seed{seed0:04d}-{seed1:04d}_i{vector_index}.png"
                )

    else:
        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print(
                "Generating image for seed %d (%d/%d) ..."
                % (seed, seed_idx, len(seeds))
            )
            z = torch.from_numpy(np.random.RandomState(seed).randn(frames, G.z_dim)).to(
                device
            )
            for vector_index in range(frames):
                vector = z[vector_index].reshape((1, -1)).to(device)
                img = G(
                    vector,
                    label,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode,
                    force_fp32=True,
                )
                img = (
                    (img.permute(0, 2, 3, 1) * 127.5 + 128)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
                    f"{outdir}/seed{seed:04d}_i{vector_index}.png"
                )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
