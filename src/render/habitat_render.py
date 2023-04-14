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
import pathlib
import shutil
import subprocess
import sys

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg

sys.path.insert(
    0, pathlib.Path(__file__).parent.joinpath("../viewer").resolve().as_posix()
)
from mocap_interface import MocapInterface


def render_video(
    sequence_folder: pathlib.Path,
    video_folder: pathlib.Path,
    sim: habitat_sim.Simulator,
    urdf_path: pathlib.Path,
):
    # Read the trajectory data
    vr_data_path = sequence_folder.joinpath("vr_data.json")
    if not vr_data_path.exists():
        return
    with open(vr_data_path) as f:
        vr_data = json.load(f)
        data = vr_data["data"]

    # Fairmotion interface
    framerate = round(1000 / data[1][0])  # The time tag for the first frame is 0
    bvh_files = list(sequence_folder.glob("*.bvh"))
    if len(bvh_files) == 0:
        return
    bvh_path = bvh_files[0].as_posix()
    start, end = vr_data["bvh_trim_indices"]
    fm = MocapInterface(
        sim,
        framerate,
        bvh_path,
        urdf_path.as_posix(),
        motion_start=start,
        motion_end=end,
    )
    assert fm.motion.num_frames() == len(data)
    fm.load_model()

    # Make video
    photo_folder = video_folder.joinpath("photos")
    if not photo_folder.exists():
        photo_folder.mkdir()

    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    sensors = [
        getattr(sensor, "uuid")
        for sensor in agent.agent_config.sensor_specifications
        if getattr(sensor, "uuid").endswith("_sensor")
    ]
    for t in range(len(data)):
        _, pos, rot = data[t]

        agent_state.position = pos
        agent_state.rotation = rot
        agent.set_state(agent_state)

        for sensor in sensors:
            sensor_type = sensor[:-7]
            observation = habitat_sim.utils.viz_utils.observation_to_image(
                sim.get_sensor_observations()[sensor],
                observation_type=sensor_type,  # trim away "_sensor"
            )

            output_path = photo_folder.joinpath(sensor + "_" + "%06d.png" % (t))
            observation.save(output_path)

        fm.next_pose()

    # Delete the model so that we can reuse the simulator for the next video
    fm.hide_model()

    suffix = {"color_sensor": "rbg", "depth_sensor": "d"}
    for sensor in sensors:
        output_vid_file = video_folder.joinpath(
            f"{bvh_files[0].stem}_{suffix[sensor]}.mp4"
        )
        command = [
            "ffmpeg",
            "-r",
            str(framerate),
            "-y",
            "-threads",
            "16",
            "-i",
            f"{photo_folder.as_posix()}/{sensor}_%06d.png",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-v",
            "error",
            output_vid_file.as_posix(),
        ]

        print(f'Running "{" ".join(command)}"')
        subprocess.call(command)

    shutil.rmtree(photo_folder.as_posix())


def main(
    sequence_folder: pathlib.Path,
    urdf_path: pathlib.Path,
    scene_dataset: str = "default",
    video_scene: str = "NONE",
):
    video_folder = sequence_folder.joinpath("habitat_rgbd")
    if not video_folder.exists():
        video_folder.mkdir()
    elif len(list(video_folder.glob("*_rgb.mp4"))) and len(
        list(video_folder.glob("*_d.mp4"))
    ):
        print(f"Skipping {sequence_folder.as_posix()}")
        return

    # Check if the user provided a URDF file
    if not urdf_path.as_posix().endswith(".urdf"):
        raise ValueError(f"Expected a URDF file: {urdf_path.as_posix()}")
    if not urdf_path.exists():
        raise ValueError(f"URDF file not found: {urdf_path.as_posix()}")

    # Initialize simulator settings
    sim_settings = default_sim_settings
    sim_settings["scene"] = video_scene
    sim_settings["scene_dataset_config_file"] = scene_dataset
    sim_settings["depth_sensor"] = True
    sim_settings["sensor_height"] = 0.0
    sim_settings["enable_physics"] = True

    # Better lighting
    cfg = make_cfg(sim_settings)
    cfg.sim_cfg.override_scene_light_defaults = True
    cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    # Start simulator
    sim = habitat_sim.Simulator(cfg)

    # Render the video
    print(f"Rendering {sequence_folder.as_posix()}")
    render_video(sequence_folder, video_folder, sim, subject)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Sequence folder to process")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--scene-dataset", default="default")
    parser.add_argument("--video-scene", default="NONE")
    args = parser.parse_args()

    sequence_folder = pathlib.Path(args.folder).expanduser()
    if not sequence_folder.exists():
        raise FileNotFoundError(f"Folder {sequence_folder} does not exist")
    subject = pathlib.Path(args.urdf).expanduser()

    main(
        sequence_folder,
        subject,
        scene_dataset=args.scene_dataset,
        video_scene=args.video_scene,
    )
