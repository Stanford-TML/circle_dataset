# CIRCLE Habitat viewer

The viewer provided in this folder can be used to preview CIRCLE sequences inside Habitat. To run it:

1. If you do not have it already, install [habitat-sim](https://github.com/facebookresearch/habitat-sim) (the [conda installation](https://github.com/facebookresearch/habitat-sim#recommended-conda-packages) is recommended).
2. Download the [scene and the URDF skeletons](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_assets.zip). Then, download the [scene without doors](https://circledataset.s3.us-west-2.amazonaws.com/release/102815835-no-doors.glb) and add it to the `stages` folder.
1. Inside this folder, run

```
python circle_viewer.py \
    --scene 102815835-no-doors \
    --dataset [path to]/floorplanner.scene_dataset_config.json \
    --experiment_dir [path to CIRCLE sequence folder] \
    --urdf [path to subject URDF]
```
