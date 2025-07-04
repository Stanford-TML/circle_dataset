"""
MIT License

Copyright (c) Meta Platforms, Inc. and its affiliates.
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

import json
import os
import pathlib
from typing import Any, Dict, List, Tuple

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration
from habitat_sim.logging import logger
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from habitat_viewer import HabitatSimInteractiveViewer, Timer
from magnum.platform.glfw import Application
from mocap_interface import MocapInterface


class MocapInteractiveViewer(HabitatSimInteractiveViewer):
    def __init__(
        self,
        sim_settings: Dict[str, Any],
        fm_settings: Dict[str, Any],
        vr_data: List,
    ) -> None:
        super().__init__(sim_settings)
        self.sim.get_debug_line_render().set_line_width(5.0)

        # fairmotion init
        start, end = vr_data["bvh_trim_indices"]
        self.fm_demo = MocapInterface(
            self.sim,
            self.fps,
            bvh_path=fm_settings["bvh_path"],
            urdf_path=fm_settings["urdf_path"],
            motion_start=start,
            motion_end=end,
        )
        self.paused = False

        # cache argument values for reconfigure
        self.fm_settings = fm_settings

        # FPOV for farimotion character
        # We cycle through the following modes when G is
        # pressed (Shift+G to cycle in the opposite
        # direction)
        #   0 = 3rd person
        #   1 = Ego4D
        #   2 = Headset trajectory replay
        self.first_person_modes = ["none", "ego4d", "headset"]
        self.first_person = 0
        self.fm_demo.load_model()

        # Skeleton
        self.head_point = None
        self.head_traj_obj = None

        self.head_link_name = "Head"
        self.head_end_link_name = "HeadEnd"
        self.ee_link_name = vr_data["target_bone"] + "End"
        self.link_name_to_id = {}

        link_name_to_id = self.link_name_to_id = {
            self.head_end_link_name: None,
            self.head_link_name: None,
            self.ee_link_name: None,
        }

        for link_id in self.fm_demo.model.get_link_ids():
            name = self.fm_demo.model.get_link_name(link_id)
            if name in link_name_to_id:
                link_name_to_id[name] = link_id

        for key, value in link_name_to_id.items():
            self.viewer_assert(value, f"Link {key} not found.")

        self.head_rot = []
        self.head_tra = []
        self.cache_skeleton_head_traj()

        # Headset
        self.headset_point = None
        self.headset_traj_obj = None
        self.cache_headset_traj(vr_data)

        # Start and goal positions
        self.start = self.fm_demo.model.get_link_scene_node(
            link_name_to_id[self.ee_link_name]
        ).translation
        self.goal = mn.Vector3(vr_data["goal"])

        self.debug_draw_enabled = False

    def cache_headset_traj(self, vr_data):
        self.viewer_assert(
            self.head_tra != [],
            "Call self.cache_skeleton_head_traj before this function.",
        )

        self.vr_traj_data = vr_data["data"]

        self.viewer_assert(
            len(self.vr_traj_data) == len(self.head_tra),
            "Mocap and VR data have different lengths.",
        )
        self.viewer_assert(
            self.vr_traj_data[0][0] == 0,
            "VR data does not start at time 0.",
        )

        self.vr_traj_times = np.cumsum([i[0] for i in self.vr_traj_data])
        self.vr_traj = [mn.Vector3(i[1]) for i in self.vr_traj_data]
        self.vr_traj_rot = [
            mn.Quaternion(mn.Vector3(i[2][:3]), i[2][3]) for i in self.vr_traj_data
        ]
        self.pose_lookup = dict(
            zip(self.vr_traj_times, zip(self.vr_traj, self.vr_traj_rot))
        )

    def cache_skeleton_head_traj(self):
        link_name_to_id = self.link_name_to_id
        head_bottom_link_name = self.head_link_name
        head_top_link_name = self.head_end_link_name

        model = self.fm_demo.model
        motion = self.fm_demo.motion

        n_frames = motion.num_frames()
        for i in range(n_frames):
            (
                new_pose,
                new_root_translate,
                new_root_rotation,
            ) = self.fm_demo.convert_CMUamass_single_pose(
                motion.poses[i], model, raw=True
            )

            model.joint_positions = new_pose
            model.rotation = new_root_rotation
            model.translation = new_root_translate

            top_scene_node = model.get_link_scene_node(
                link_name_to_id[head_top_link_name]
            )
            bottom_scene_node = model.get_link_scene_node(
                link_name_to_id[head_bottom_link_name]
            )
            self.head_rot.append(bottom_scene_node.rotation)
            self.head_tra.append(top_scene_node.translation)

        # Reset the model to the pose we were in before calling this function
        self.fm_demo.next_pose(repeat=True)

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key press by performing the corresponding functions.
        If the key pressed is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the
        key will be set to False for the next `self.move_and_look()` to update the current actions.
        """

        key = event.key
        pressed = Application.Key
        mod = Application.Modifier

        shift_pressed = bool(event.modifiers & mod.SHIFT)

        super().key_press_event(event)

        if key == pressed.U:
            self.debug_draw_enabled = not self.debug_draw_enabled

        if key == pressed.F:
            self.viewer_assert(self.fm_demo.model, "Model has not been loaded")
            self.draw_vr_traj()
            self.draw_bvh_traj()

        elif key == pressed.G:
            # Toggle fpov. Pressing shift goes back
            self.first_person = (self.first_person + (-1) ** shift_pressed) % len(
                self.first_person_modes
            )
            logger.info(
                f"Command: set FPOV to {self.first_person_modes[self.first_person]}"
            )

        elif key == pressed.I:
            # Pause the skeleton animation
            self.paused = not self.paused

        elif key == pressed.Q:
            photo_dir = pathlib.Path(os.getcwd())
            observation = habitat_sim.utils.viz_utils.observation_to_image(
                self.sim.get_sensor_observations()["color_sensor"],
                observation_type="color",
            )

            output_path = photo_dir.joinpath("screenshot.png")
            observation.save(output_path)

    def debug_draw(self):
        super().debug_draw()

        if not self.debug_draw_enabled:
            return

        self.draw_frame(self.start, mn.Quaternion().to_matrix())
        self.draw_frame(self.goal, mn.Quaternion().to_matrix())

        self.sim.get_debug_line_render().draw_transformed_line(
            self.start, self.goal, mn.Color4(1.0, 0.0, 0.0, 1.0)
        )

        if self.headset_point:
            self.draw_frame(self.headset_point, self.headset_point_rot.to_matrix())
            diagonal = mn.Vector3(0.1, 0.1, 0.1)
            self.sim.get_debug_line_render().push_transform(
                mn.Matrix4.from_(self.headset_point_rot.to_matrix(), self.headset_point)
            )
            self.sim.get_debug_line_render().draw_box(
                diagonal,
                -diagonal,
                mn.Color4(0.0, 1.0, 0.0, 1.0),
            )
            self.sim.get_debug_line_render().pop_transform()

        if self.head_point:
            self.draw_frame(self.head_point, self.head_point_rot.to_matrix())

    def draw_frame(self, root, rot):
        self.sim.get_debug_line_render().draw_transformed_line(
            root - 0.1 * rot[0], root + 0.1 * rot[0], mn.Color4(1.0, 0.0, 0.0, 1.0)
        )
        self.sim.get_debug_line_render().draw_transformed_line(
            root - 0.1 * rot[1], root + 0.1 * rot[1], mn.Color4(0.0, 1.0, 0.0, 1.0)
        )
        self.sim.get_debug_line_render().draw_transformed_line(
            root - 0.1 * rot[2], root + 0.1 * rot[2], mn.Color4(0.0, 0.0, 1.0, 1.0)
        )

    def draw_vr_traj(self):
        if self.headset_traj_obj:
            # Todo: Figure out how to hide the trajectories
            return

        self.headset_traj_obj = self.sim.add_trajectory_object(
            "HeadsetTraj",
            self.vr_traj,
            color=mn.Color4([0.1, 0.9, 0.1, 1.0]),
            radius=0.005,
        )

    def draw_bvh_traj(self):
        if self.head_traj_obj:
            # Todo: Figure out how to hide the trajectories
            return

        self.head_traj_obj = self.sim.add_trajectory_object(
            "HeadTraj",
            self.head_tra,
            color=mn.Color4([0.1, 0.1, 0.9, 1.0]),
            radius=0.005,
        )

    def draw_event(
        self, active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor")
    ) -> None:
        def play_motion() -> None:
            if self.paused:
                return

            self.fm_demo.next_pose()

            self.headset_point, self.headset_point_rot = self.pose_lookup[
                (self.fm_demo.motion_stepper) * (1000 / self.fm_demo.motion.fps)
            ]

            # Update the headset pose
            top_scene_node = self.fm_demo.model.get_link_scene_node(
                self.link_name_to_id[self.head_end_link_name]
            )
            self.head_point = top_scene_node.translation
            self.head_point_rot = top_scene_node.rotation

        def run_global() -> None:
            # choose agent
            if (
                self.first_person
                and self.fpov_agent
                and self.fm_demo
                and self.fm_demo.model
            ):
                self.render_first_person_pov()

        # choose agent
        if (
            self.first_person
            and self.fpov_agent
            and self.fm_demo
            and self.fm_demo.model
        ):
            keys = (self.fpov_agent_id, "fpov_sensor")
        else:
            keys = active_agent_id_and_sensor_name

        # Run the parent method
        super().draw_event(
            simulation_call=play_motion,
            global_call=run_global,
            active_agent_id_and_sensor_name=keys,
        )

    def reconfigure_sim(self) -> None:
        """
        Utilizes the current `self.sim_settings` to configure and set up a new
        `habitat_sim.Simulator`, and then either starts a simulation instance, or replaces
        the current simulator instance, reloading the most recently loaded scene
        """
        # configure our sim_settings but then set the agent to our default
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        if self.enable_batch_renderer:
            self.cfg.enable_batch_renderer = True
            self.cfg.sim_cfg.create_renderer = False
            self.cfg.sim_cfg.enable_gfx_replay_save = True

        if self.sim_settings["stage_requires_lighting"]:
            logger.info("Setting synthetic lighting override for stage.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        # first person agent
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = mn.Vector2i(
            int(self.sim_settings["width"]),
            int(self.sim_settings["height"]),
        )
        camera_sensor_spec.position = mn.Vector3(0, 0, 0)
        camera_sensor_spec.orientation = mn.Vector3(0, 0, 0)
        camera_sensor_spec.uuid = "fpov_sensor"

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=0.01,
            radius=0.01,
            sensor_specifications=[camera_sensor_spec],
            body_type="cylinder",
        )
        self.fpov_agent_id = len(self.cfg.agents)
        self.cfg.agents.append(agent_config)

        if self.sim is None:
            self.tiled_sims = []
            for _i in range(self.num_env):
                self.tiled_sims.append(habitat_sim.Simulator(self.cfg))
            self.sim = self.tiled_sims[0]
        else:  # edge case
            for i in range(self.num_env):
                if (
                    self.tiled_sims[i].config.sim_cfg.scene_id
                    == self.cfg.sim_cfg.scene_id
                ):
                    # we need to force a reset, so change the internal config scene name
                    self.tiled_sims[i].config.sim_cfg.scene_id = "NONE"
                self.tiled_sims[i].reconfigure(self.cfg)

        # post reconfigure
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.render_camera = self.default_agent.scene_node.node_sensor_suite.get(
            "color_sensor"
        )

        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

        # Initialize replay renderer
        if self.enable_batch_renderer and self.replay_renderer is None:
            self.replay_renderer_cfg = ReplayRendererConfiguration()
            self.replay_renderer_cfg.num_environments = self.num_env
            self.replay_renderer_cfg.standalone = (
                False  # Context is owned by the GLFW window
            )
            self.replay_renderer_cfg.sensor_specifications = self.cfg.agents[
                self.agent_id
            ].sensor_specifications
            self.replay_renderer_cfg.gpu_device_id = self.cfg.sim_cfg.gpu_device_id
            self.replay_renderer_cfg.force_separate_semantic_scene_graph = False
            self.replay_renderer_cfg.leave_context_with_background_renderer = False
            self.replay_renderer = ReplayRenderer.create_batch_replay_renderer(
                self.replay_renderer_cfg
            )
            # Pre-load composite files
            if sim_settings["composite_files"] is not None:
                for composite_file in sim_settings["composite_files"]:
                    self.replay_renderer.preload_file(composite_file)

        self.agent_body_node = self.default_agent.scene_node
        self.agent_body_node.translation = mn.Vector3(1.0, 0, 3)

        self.fpov_agent = self.sim.get_agent(self.fpov_agent_id)
        self.fpov_init_rotation = (
            self.fpov_agent.scene_node.rotation
        )

        Timer.start()
        self.step = -1

    def render_first_person_pov(self) -> None:
        """
        Utilizes the first person agent to render an egocentric perspective of
        the fairmotion character upon toggle FPOV toggle.
        """
        model = self.fm_demo.model
        agent = self.fpov_agent

        if self.first_person_modes[self.first_person] == "none":
            return
        elif self.first_person_modes[self.first_person] == "ego4d":
            fw_axis = mn.Vector3.y_axis()
            up_axis = mn.Vector3.z_axis()

            head_id = [
                x
                for x in model.get_link_ids()
                if model.get_link_name(x) == self.head_link_name
            ][0]

            head_ScNode = model.get_link_scene_node(head_id)

            fw_axis = head_ScNode.transformation_matrix().transform_vector(fw_axis)
            up_axis = head_ScNode.transformation_matrix().transform_vector(up_axis)

            # applying scenenode rotation and translation to FPOV
            agent.scene_node.translation = head_ScNode.absolute_translation
            agent.scene_node.rotation = mn.Quaternion.from_matrix(
                mn.Matrix4.look_at(
                    head_ScNode.absolute_translation,
                    head_ScNode.absolute_translation + fw_axis,
                    up_axis,
                ).rotation()
            )
        elif self.first_person_modes[self.first_person] == "headset":
            agent.scene_node.translation, agent.scene_node.rotation = (
                self.headset_point,
                self.headset_point_rot,
            )

    def print_help_text(self) -> None:
        """
        Print the Key Command help text.
        """
        logger.info(
            """
=====================================================
Welcome to the Habitat-sim Python Viewer application!
=====================================================
Mouse Functions ('m' to toggle mode):
----------------
In LOOK mode (default):
    LEFT:
        Click and drag to rotate the agent and look up/down.
    WHEEL:
        Modify orthographic camera zoom/perspective camera FOV (+SHIFT for fine grained control)

In GRAB mode (with 'enable-physics'):
    LEFT:
        Click and drag to pickup and move an object with a point-to-point constraint (e.g. ball joint).
    RIGHT:
        Click and drag to pickup and move an object with a fixed frame constraint.
    WHEEL (with picked object):
        default - Pull gripped object closer or push it away.
        (+ALT) rotate object fixed constraint frame (yaw)
        (+CTRL) rotate object fixed constraint frame (pitch)
        (+ALT+CTRL) rotate object fixed constraint frame (roll)
        (+SHIFT) amplify scroll magnitude


Key Commands:
-------------
    esc:        Exit the application.
    'h':        Display this help message.
    'm':        Cycle mouse interaction modes.
    'g':        Cycle camera perspectives (3rd person, head bone, VR headset)
                (+SHIFT) Cycle in the opposite direction.

    Agent Controls:
    'wasd':     Move the agent's body forward/backward and left/right.
    'zx':       Move the agent's body up/down.
    arrow keys: Turn the agent's body left/right and camera look up/down.

    Utilities:
    'f':        Draw the head bone and headset trajectories.
    'u':        Enable debug drawing (headset representation, head bone frame of reference).
    'i':        Pause/continue the skeleton animation.
    'q':        Save a screenshot of the current display to the current folder.
    'r':        Reset the simulator with the most recently loaded scene.
    'n':        Show/hide NavMesh wireframe.
                (+SHIFT) Recompute NavMesh with default settings.
                (+ALT) Re-sample the agent(camera)'s position and orientation from the NavMesh.
    ',':        Render a Bullet collision shape debug wireframe overlay (white=active, green=sleeping, blue=wants sleeping, red=can't sleep).
    'c':        Run a discrete collision detection pass and render a debug wireframe overlay showing active contact points and normals (yellow=fixed length normals, red=collision distances).
                (+SHIFT) Toggle the contact point debug render overlay on/off.

    Object Interactions:
    SPACE:      Toggle physics simulation on/off.
    '.':        Take a single simulation step if not simulating continuously.
    'v':        (physics) Invert gravity.
    't':        Load URDF from filepath
                (+SHIFT) quick re-load the previously specified URDF
                (+ALT) load the URDF with fixed base
=====================================================
"""
        )

    def viewer_assert(self, condition, message):
        """
        Using the assert statement can cause the app to hang.
        Use this function instead.
        """
        if not condition:
            print(f"AssertionError: {message}")
            self.sim.close()
            self.exit_event(Application.ExitEvent)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="NONE",
        type=str,
        help='Scene/stage file to load (default: "NONE")',
    )
    parser.add_argument(
        "--dataset",
        default="default",
        type=str,
        metavar="DATASET",
        help="Dataset configuration file to use (default: default)",
    )
    parser.add_argument(
        "--disable_physics",
        action="store_true",
        help="Disable physics simulation (default: False)",
    )
    parser.add_argument(
        "--stage_requires_lighting",
        action="store_true",
        help="Override configured lighting to use synthetic lighting for the stage.",
    )
    parser.add_argument(
        "--enable-batch-renderer",
        action="store_true",
        help="Enable batch rendering mode. The number of concurrent environments is specified with the num-environments parameter.",
    )
    parser.add_argument(
        "--num-environments",
        default=1,
        type=int,
        help="Number of concurrent environments to batch render. Note that only the first environment simulates physics and can be controlled.",
    )
    parser.add_argument(
        "--composite-files",
        type=str,
        nargs="*",
        help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",
    )
    parser.add_argument(
        "--width",
        default=800,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="Path to the folder with the bvh and vr_data.json files",
        required=True
    )
    parser.add_argument(
        "--urdf",
        type=str,
        help="Path to subject urdf file (if not specified then it will search for one in experiment_dir)",
    )

    args = parser.parse_args()

    experiment_dir = pathlib.Path(args.experiment_dir)
    if args.urdf:
        urdf_path = args.urdf
    else:
        urdf_candidates = list(experiment_dir.glob("*.urdf"))
        if len(urdf_candidates) == 0:
            raise argparse.ArgumentError("urdf", "Please specify path to URDF file.")
        urdf_path = urdf_candidates[0].as_posix()
    bvh_path = list(experiment_dir.glob("*.bvh"))[0].as_posix()
    with open(experiment_dir.joinpath("vr_data.json")) as f:
        vr_data = json.load(f)

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["stage_requires_lighting"] = args.stage_requires_lighting
    sim_settings["enable_batch_renderer"] = args.enable_batch_renderer
    sim_settings["num_environments"] = args.num_environments
    sim_settings["composite_files"] = args.composite_files
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height

    fm_settings: Dict[str, Any] = {}
    fm_settings["bvh_path"] = bvh_path
    fm_settings["urdf_path"] = urdf_path

    MocapInteractiveViewer(sim_settings, fm_settings, vr_data).exec()
