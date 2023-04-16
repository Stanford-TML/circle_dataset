# CIRCLE: Capture In Rich Contextual Environments (CVPR 2023)

This repo hosts the website for CIRCLE dataset.

## Download

The CIRCLE data are hosted in an AWS S3 bucket. We provide the motion data separate from the first person videos. Please use the following links to download:

* [Motion (SMPL-X and BVH) and headset trajectories](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_movement.zip)
* [Habitat first person videos](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_habitat_videos.zip)
* [Blender first person videos](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_blender_videos.zip)
* [Scene and subject URDF files](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_assets.zip)

If you find any issues with the dataset, please [let us know](https://github.com/Stanford-TML/circle_dataset/issues/new).

CIRCLE data are licensed under a [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. For the license of the code in this repository, see the [LICENSE](https://github.com/Stanford-TML/circle_dataset/blob/main/LICENSE) file.

## Visualization

The recommended way to visualize CIRCLE data is to use [Blender](https://www.blender.org/). All sequences are stored assuming a `Y up, -Z forward` frame of reference. Below we describe the steps required to make sure the sequences are loaded in the correct orientation.

### Visualizing using Blender.

Use the Blender import menu to import the scene `.glb` file. To visualize the motion, either use `File -> Import -> Motion capture (.bvh)` (make sure to enable `Scale FPS` and `Update Scene Duration`), or download and install the [SMPL-X Blender add-on](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon), and follow the instructions below:

1. On the 3D viewport, press `N` to show the sidebar.
2. Click on the `SMPL-X` separator.
1. Click `Add Animation`. Set the format to `SMPL-X` and load the `.npz` file of the sequence you would like to inspect.

### Visualizing using Habitat.

We also provide a [custom habitat-sim viewer](https://github.com/Stanford-TML/circle_dataset/tree/main/src/viewer), which exemplifies how to load CIRCLE sequences into Habitat (check the [`__init__` method](https://github.com/Stanford-TML/circle_dataset/blob/main/src/viewer/circle_viewer.py#L53-L61) of `circle_viewer.py` and the [`next_pose` method](https://github.com/Stanford-TML/circle_dataset/blob/main/src/viewer/mocap_interface.py#L140-L150) in `mocap_interface.py`).

## Rendering

To reproduce the first person videos distributed with CIRCLE, check the [`render` folder](https://github.com/Stanford-TML/circle_dataset/tree/main/src/render).
