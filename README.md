# CIRCLE: Capture In Rich Contextual Environments (CVPR 2023)

This repo hosts the website for CIRCLE dataset.

## UPDATE (10-01-2026):

The CIRCLE assets and movement data files are now hosted on a different platform with a lower storage capacity. As such, first person videos are no longer provided. They can be generated again using an adapted version of the provided code.

## Download

The CIRCLE data are hosted in an AWS S3 bucket. We provide the motion data separate from the first person videos. Please use the following links to download:

* [Motion (SMPL-X and BVH) and headset trajectories](https://drive.google.com/file/d/1Zq3u8bHawtxrFeKgpiOU7bm5yJBemHWW/view?usp=sharing)
* ~Habitat first person videos~
* ~Blender first person videos~
* [Scene and subject URDF files](https://drive.google.com/file/d/1NACAR1l90x6v8ibJCSPR7EFUGNdFY0SK/view?usp=sharing)
    * [Scene without doors](https://drive.google.com/file/d/1sOJycxrcy_VUEy-PT5WRyX7Nzg_XbSZI/view?usp=sharing)

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
