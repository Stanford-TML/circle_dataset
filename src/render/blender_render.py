"""
MIT License

Copyright (c) 2023 The Movement Lab @ Stanford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

import imageio
import numpy as np
import torch
import trimesh
from human_body_prior.body_model.body_model import BodyModel


def images_to_video(img_folder, output_vid_file):
    img_files = os.listdir(img_folder)
    img_files.sort()
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        try:
            im = imageio.imread(img_path)
            im_arr.append(im)
        except:
            print(f"Skipping file {img_name}")

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8)


def save_objs(sequence_folder: pathlib.Path, out_folder: pathlib.Path):
    npz_files = list(sequence_folder.glob("*.npz"))
    if len(npz_files) == 0:
        print("No .npz files found")
        return
    elif len(npz_files) > 1:
        print("More than one .npz file found.")
        return

    npz_file = npz_files[0].as_posix()
    data = np.load(npz_file)

    if "SMPLX_MODEL_PATH" not in os.environ:
        raise KeyError(
            'Please set environment variable "SMPLX_MODEL_PATH" before running this script.'
        )
    support_base_dir = os.environ.get("SMPLX_MODEL_PATH")
    gender = str(data["gender"])
    surface_model_fname = os.path.join(support_base_dir, gender, "model.npz")
    mocap_frame_rate = data["mocap_frame_rate"]
    target_frame_rate = 30

    step_size = int(mocap_frame_rate / target_frame_rate)

    bm = BodyModel(
        bm_fname=surface_model_fname,
        num_betas=data["betas"].shape[0],
        num_expressions=0,
    )

    shaped_data = torch.from_numpy(data["poses"]).float()
    N = shaped_data.shape[0]
    root_orient = shaped_data[:, 0:3]
    pose_body = shaped_data[:, 3:66]
    pose_hand = shaped_data[:, -90:]
    out = bm(
        pose_body=pose_body,
        pose_hand=pose_hand,
        betas=torch.from_numpy(data["betas"]).float().unsqueeze(0).expand(N, -1),
        root_orient=root_orient,
        trans=torch.from_numpy(data["trans"]).float(),
    )
    mesh_verts = out.v
    mesh_faces = out.f

    save_mesh_folder = out_folder.joinpath("meshes")
    if not save_mesh_folder.exists():
        save_mesh_folder.mkdir()

    num_meshes = mesh_verts.shape[0]
    for idx in range(0, num_meshes, step_size):
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d" % (idx) + ".obj")
        if os.path.exists(curr_mesh_path):
            continue
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx], faces=mesh_faces)
        mesh.export(curr_mesh_path)

    return save_mesh_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        metavar="PATH",
        help="path to a CIRCLE sequence folder",
        default="",
    )
    parser.add_argument(
        "--scene",
        type=str,
        metavar="PATH",
        help="path to specific .blend path for 3D scene",
        default=pathlib.Path(__file__).parent.parent.joinpath(
            "data", "blender_fpv.blend"
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        metavar="PATH",
        help="path to stage .glb file",
        default="",
    )
    args = parser.parse_args()

    sequence_folder = pathlib.Path(args.folder)
    out_folder_path = sequence_folder.joinpath("blender_rgbd")

    # Save the obj files
    mesh_folder = save_objs(sequence_folder, out_folder_path)
    if not mesh_folder:
        print("Failed to create mesh folder")
        quit()

    img_folder = out_folder_path.joinpath("imgs")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    utils_path = pathlib.Path(__file__).parent.joinpath("blender_render_utils.py")
    depth_file = img_folder.joinpath("depth.npy")

    # Run Blender
    if "darwin" in sys.platform:
        blender = "/Applications/Blender.app/Contents/MacOS/Blender"
    else:
        blender = "blender"
    subprocess.call(
        " ".join(
            [
                blender,
                "-P",
                utils_path.as_posix(),
                "-b",
                "--",
                f"--sequence-folder {sequence_folder.as_posix()}",
                f"--mesh-folder {mesh_folder.as_posix()}",
                f"--scene {args.scene}",
                f"--out-folder {img_folder.as_posix()}",
                f"--stage {args.stage}",
            ]
        ),
        shell=True,
    )

    out_vid_path = out_folder_path.joinpath(
        sequence_folder.name + "_rgb.mp4"
    ).as_posix()
    images_to_video(img_folder, out_vid_path)

    final_depth_file = out_folder_path.joinpath(sequence_folder.name + "_d.npy")
    shutil.move(depth_file, final_depth_file)
    data = np.load(final_depth_file.as_posix())
    if data.dtype != np.float16:
        data = data.astype(np.float16)
    # Prepare to save the depth buffer as a video
    # Set the values at infinity to zero
    data[data == np.inf] = 0
    # Normalize from 0 to 10 (far clipping plane) to 0 to 1
    data /= 10
    # Convert from 0 to 1 float to 0 to 255 integer
    data = (data * 255).astype(np.uint8)
    # Save the depth buffer as a video
    blender_d_vid = out_folder_path.joinpath(
        out_folder_path.joinpath(sequence_folder.name + "_d.mp4")
    )
    imageio.mimwrite(blender_d_vid.as_posix(), data, fps=30, quality=8)

    # Clean up
    shutil.rmtree(img_folder.as_posix())
    if mesh_folder:
        shutil.rmtree(mesh_folder.as_posix())
