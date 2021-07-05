import os
import io
import sys
import subprocess

import click
import time

# import torch
from flask import Flask, jsonify, request, Response
import requests
from argparse import Namespace
from pathlib import Path

# change path inside stylegan2 repo
import sys

# sys.path.append("./stylegan2-pytorch")
import numpy as np
import torch
from PIL import Image
import legacy
import dnnlib
from typing import List, Optional

global ckpt_dir
ckpt_dir = "./checkpoint"

app = Flask(__name__)
app.config["SERVER_NAME"] = os.environ.get("SERVER_NAME")


# mjpeg functions
class Camera(object):
    def __init__(self, generator, walk_vectors, class_idx, psi, noise_mode, device):
        self.generator = generator
        self.walk_vectors = walk_vectors
        # self.frames = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]
        self.walk_index = 0
        self.vector_index = 0

        # Labels.
        self.label = torch.zeros([1, generator.c_dim], device=device)
        if generator.c_dim != 0:
            self.label[:, class_idx] = 1
        self.psi = psi
        self.noise_mode = noise_mode
        self.device = device

    def get_frame(self):
        # return self.frames[int(time()) % 3]
        print("start generate image", self.walk_index, self.vector_index)
        latent_z = self.walk_vectors[self.walk_index].astype("float32")
        vector = latent_z[self.vector_index].reshape((1, -1))
        vector = torch.tensor(vector).to(self.device)
        img = self.generator(
            vector,
            self.label,
            truncation_psi=self.psi,
            noise_mode=self.noise_mode,
            force_fp32=True,
        )
        # converting image to uint8
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_image = img[0].cpu().numpy()
        # save image
        im = Image.fromarray(out_image)
        tempImage = io.BytesIO()
        im.save(tempImage, format="JPEG")

        # update walk_index and vector_index
        self.vector_index = (self.vector_index + 1) % latent_z.shape[0]
        if self.vector_index == 0:
            # update walk_index
            self.walk_index = (self.walk_index + 1) % len(self.walk_vectors)
        print("return image", self.walk_index, self.vector_index)
        # load jpg in binary
        # im = open("./temp.jpeg", "rb").read()
        return tempImage.getvalue()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


pretrained = {
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    "metfaces": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
    "afhqcat": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl",
    "afhqdog": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
    "afhqwild": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl",
    "cifar10": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl",
    "brecahad": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/brecahad.pkl",
}


@app.route("/generate-latent-walk", methods=["get", "post"])
def generate():
    # req = request.get_json(force=True)
    # print("req", req)
    # ckpt_file = req.get("ckpt_file", "ffhq")
    # seeds = req.get("seeds", "3,7")
    # frames = int(req.get("frames", 10))
    # psi = float(req.get("psi", 1.0))
    # class_idx = int(req.get("class_idx", 0))
    # noise_mode = req.get("noise_mode", "const")

    ckpt_file = "model.pkl"
    seeds = "3,7,10"
    frames = 10
    psi = 1.0
    class_idx = 0
    noise_mode = "const"

    # validate seeds
    seeds = seeds.split(",")
    seeds = [int(s) for s in seeds]
    # # validate ckpt_file

    device = torch.device("cpu")
    if ckpt_file in pretrained.keys():
        network_pkl = pretrained[ckpt_file]

        # load model
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    else:
        ckpt_file_name = str(Path(ckpt_file).stem)

        ckptfile_list = Path(ckpt_dir).rglob("*.*")
        target_ckpt = None
        for ckpt in ckptfile_list:
            print("file:", ckpt)
            ckpt_name = str(Path(ckpt).stem)
            if ckpt_name == ckpt_file_name:
                target_ckpt = str(ckpt)
                break

        if target_ckpt is None:
            raise Exception(f"ckpt file:{ckpt_file} not found in ckpt dir:{ckpt_dir}")

        # load model
        print('Loading networks from "%s"...' % target_ckpt)
        with dnnlib.util.open_url(target_ckpt) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    ##############

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

    # load jpeg to build mjpeg stream
    return Response(
        gen(Camera(G, walk_vectors, class_idx, psi, noise_mode, device)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status", methods=["GET"])
def status():
    return "ok"


def setup(cli_checkpoint_dir="./checkpoint/"):
    global checkpoint_dir
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR") or cli_checkpoint_dir
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return app


@click.command()
@click.option("--debug", "-d", is_flag=True)
@click.option("--checkpoint-dir", "-cp", default="./checkpoint/")
def api_run(debug, checkpoint_dir):
    app = setup(checkpoint_dir)
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    api_run()
