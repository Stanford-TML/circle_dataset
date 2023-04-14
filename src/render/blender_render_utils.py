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
import json
import os
import pathlib
import shutil

import numpy as np

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

CORRECTION = Matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
CLIP_MAX = 10
RES_X = 640
RES_Y = 480


def create_nodes(output_dir):
    # Setup
    bpy.data.scenes["Scene"].use_nodes = True
    bpy.data.scenes["Scene"].view_layers["View Layer"].use_pass_combined = False
    bpy.data.scenes["Scene"].view_layers["View Layer"].use_pass_z = True
    bpy.data.scenes["Scene"].view_layers["View Layer"].use_pass_mist = False
    tree = bpy.context.scene.node_tree

    # Get the default nodes
    render_layers = tree.nodes["Render Layers"]

    # Add nodes
    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.use_alpha = False

    depth_file_out = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_out.base_path = os.path.join(output_dir, "tmp_depth")
    depth_file_out.format.file_format = "OPEN_EXR"

    # Linking
    links = tree.links
    links.new(
        render_layers.outputs["Depth"], depth_file_out.inputs[0]
    )  # link Z to output


def init_blender(args):
    # Load the world
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    # Set the framerate
    bpy.data.scenes["Scene"].render.fps = 30

    # Set the FPV camera parameters
    bpy.data.cameras["Camera"].lens_unit = "MILLIMETERS"
    bpy.data.cameras["Camera"].lens = 15
    bpy.data.cameras["Camera"].clip_start = 0.01
    bpy.data.cameras["Camera"].clip_end = CLIP_MAX

    bpy.data.scenes["Scene"].render.resolution_x = RES_X
    bpy.data.scenes["Scene"].render.resolution_y = RES_Y

    # Set the camera rotation mode
    camera_object = bpy.data.objects["Camera"]
    camera_object.rotation_mode = "QUATERNION"

    # Load the stage
    stage_path = pathlib.Path(args.stage)
    bpy.ops.import_scene.gltf(
        filepath=stage_path.as_posix(),
        files=[{"name": stage_path.name}],
        loglevel=50,
    )
    # Make the ceiling cast no shadow
    bpy.data.objects["geometry_ceiling"].visible_shadow = False

    # Same for the walls
    children = []
    for i in bpy.data.objects["walls"].children:
        children.append(i)

    while children:
        el = children.pop()
        if el.children:
            for i in el.children:
                children.append(i)
        el.visible_shadow = False

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    device_types = [
        i[0]
        for i in bpy.context.preferences.addons["cycles"].preferences.get_device_types(
            bpy.context
        )
    ]
    if "CUDA" in device_types:
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
    elif "METAL" in device_types:
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "METAL"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])


def set_camera_pose(camera_object, frame, correction):
    loc = Vector(frame[1]) @ correction
    camera_object.location = loc
    rot = (
        correction.transposed() @ Quaternion([frame[2][-1]] + frame[2][0:3]).to_matrix()
    )
    camera_object.rotation_quaternion = rot.to_quaternion()


def render_ply_folder(obj_folder, output_dir, vr_data):
    # Load the head vertex ids
    head_idxs = np.load(
        pathlib.Path(__file__)
        .parent.parent.joinpath("data", "head_vert_idx.npy")
        .as_posix()
    )

    # Prepare ply paths
    ori_obj_files = os.listdir(obj_folder)
    ori_obj_files.sort()
    obj_files = []
    for tmp_name in ori_obj_files:
        if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            obj_files.append(tmp_name)

    # Set up material
    human_mat = bpy.data.materials.new(
        name="MaterialName"
    )  # set new material to variable
    human_mat.use_nodes = True
    principled_bsdf = human_mat.node_tree.nodes["Principled BSDF"]
    if principled_bsdf is not None:
        # Light Blue
        principled_bsdf.inputs[0].default_value = (
            10 / 255.0,
            30 / 255.0,
            225 / 255.0,
            1,
        )
    human_mat.use_backface_culling = True

    # Prepare the depth buffer
    depth = np.zeros(
        (
            min(len(obj_files), len(vr_data)),  # These ones are already set to 30 FPS
            RES_Y,
            RES_X,
        )
    )

    # Render each frame
    camera_object = bpy.data.objects["Camera"]
    for frame_idx in range(min(len(obj_files), len(vr_data))):
        bpy.context.scene.frame_current = frame_idx + 1

        file_name = obj_files[frame_idx]

        # Iterate folder to process all model
        path_to_file = os.path.join(obj_folder, file_name)

        # RGD frame_idx will be off by 1 relative to Blender
        rgb_image_name = ("%05d" % frame_idx) + ".jpg"
        out_rgb_file = os.path.join(output_dir, rgb_image_name)

        exr_image_name = "Image" + ("%04d" % (frame_idx + 1)) + ".exr"
        out_exr_file = os.path.join(output_dir, "tmp_depth", exr_image_name)

        if not (os.path.exists(out_rgb_file) and os.path.exists(out_exr_file)):
            # Load human mesh and set material
            if ".obj" in path_to_file:
                bpy.ops.wm.obj_import(filepath=path_to_file)
            elif ".ply" in path_to_file:
                bpy.ops.import_mesh.ply(filepath=path_to_file)

            human_obj_object = bpy.data.objects[
                str(file_name.replace(".ply", "").replace(".obj", ""))
            ]
            human_mesh = human_obj_object.data
            for f in human_mesh.polygons:
                f.use_smooth = True

            human_obj_object.data.materials.append(human_mat)
            human_obj_object.active_material = human_mat

            # Select and delete the head
            bpy.ops.object.editmode_toggle()
            me = human_obj_object.data
            bm = bmesh.from_edit_mesh(me)
            for v in bm.verts:
                v.select = False
                if v.index in head_idxs:
                    v.select = True
            bmesh.update_edit_mesh(me)
            bpy.ops.mesh.delete(type="VERT")
            bpy.ops.object.editmode_toggle()

            # Render 1st person view
            set_camera_pose(camera_object, vr_data[frame_idx], CORRECTION)

            bpy.data.scenes["Scene"].render.filepath = out_rgb_file
            bpy.ops.render.render(write_still=True)

            bpy.data.objects.remove(human_obj_object, do_unlink=True)

        else:
            print(f"Skipping frame {frame_idx+1}")

        # Get the depth buffer
        bpy.ops.image.open(filepath=out_exr_file)
        z_pixels = np.array(bpy.data.images[exr_image_name].pixels)[::4]
        depth[frame_idx] = np.flipud(z_pixels.reshape((RES_Y, RES_X)))

    # Save the depth buffer
    np.save(os.path.join(output_dir, "depth.npy"), depth.astype(np.float16))
    shutil.rmtree(os.path.join(output_dir, "tmp_depth"))

    # Delete materials
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description="Render Motion in 3D Environment.")
    parser.add_argument(
        "--sequence-folder",
        type=str,
        metavar="PATH",
        help="path to CIRCLE sequence folder",
        default="",
    )
    parser.add_argument(
        "--mesh-folder",
        type=str,
        metavar="PATH",
        help="path to folder containing the sequence .obj files",
        default="",
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        metavar="PATH",
        help="path to output folder which include rendered img files",
        default="",
    )
    parser.add_argument(
        "--scene",
        type=str,
        metavar="PATH",
        help="path to specific .blend path for 3D scene",
        default="",
    )
    parser.add_argument(
        "--stage",
        type=str,
        metavar="PATH",
        help="path to stage .glb file",
        default="",
    )
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Init and configure Blender
    init_blender(args)

    # Create the nodes for 1st person view
    create_nodes(args.out_folder)

    scene_name = pathlib.Path(args.scene).stem
    print("scene name:{0}".format(scene_name))

    output_dir = args.out_folder
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the headset trajectory data
    with open(os.path.join(args.sequence_folder, "vr_data.json")) as f:
        # VR data is stored at 120 fps, so we load every 4 frames
        vr_data = [frame for frame in json.load(f)["data"][::4]]

    print("obj_folder:{0}".format(args.mesh_folder))
    render_ply_folder(args.mesh_folder, output_dir, vr_data)

    bpy.ops.wm.quit_blender()
