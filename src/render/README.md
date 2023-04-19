# CIRCLE render scripts

The scripts in this folder can be used to re-render the videos that are distributed as part of CIRCLE.

Note that the final CIRCLE renderings used a modified version of the scene without the doors and the windows. This version can be recreated by deleting the `openings` collection from the scene `.glb` file using Blender. Alternatively, you can download it [here](https://circledataset.s3.us-west-2.amazonaws.com/release/102815835-no-doors.glb). The videos being distributed as part of CIRCLE were also compressed using [Handbrake](https://handbrake.fr/).

## Habitat render

1. Make sure habitat-sim is installed and `ffmpeg` is in your path.
2. Call `python habitat_render.py --folder [path to CIRCLE sequence folder] --urdf [path to subject URDF file] --scene-dataset [path to floorplanner.scene_dataset_config.json] --video-scene 102815835` (or `--video-scene 102815835-no-doors`)

## Blender render

1. Make sure the [Blender](https://www.blender.org) executable is in your path (on MacOS make sure it is in the `Applications` folder).
2. Download the SMPL-X models and put them on a folder like this (use symlinks if you already have the SMPL-X models downloaded)

```
    smplx
    ├── male
    │   └── model.npz
    ├── female
    │   └── model.npz
    └── neutral
        └── model.npz
```

3. Set an environment variable called `SMPLX_MODEL_PATH` to be the path to the `smplx` folder from the previous step.
4. Run `pip install trimesh imageio torch human-body-prior` to make sure you have all the dependencies.
5. Call `python blender_render.py --folder [path to CIRCLE sequence folder] --stage [path to the .glb file]`. We provide a Blender file with default lighting. Use the `--scene` option to specify the path to a Blender file with a custom lighting setup.

Running this script also saves the depth buffer to an `.npy` file. Note that these files tend to be quite large (>100MB).
