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

from typing import List, Optional, Tuple

import habitat_sim
import habitat_sim.physics as phy
import magnum as mn
from fairmotion.core import motion
from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.ops.motion import cut
from habitat_sim.logging import logger


def global_correction_quat(up_v: mn.Vector3, forward_v: mn.Vector3) -> mn.Quaternion:
    """
    Given the upward direction and the forward direction of a local space frame, this methd produces
    the correction quaternion to convert the frame to global space (+Y up, -Z forward).
    """
    if up_v.normalized() != mn.Vector3.y_axis():
        angle1 = mn.math.angle(up_v.normalized(), mn.Vector3.y_axis())
        axis1 = mn.math.cross(up_v.normalized(), mn.Vector3.y_axis())
        rotation1 = mn.Quaternion.rotation(angle1, axis1)
        forward_v = rotation1.transform_vector(forward_v)
    else:
        rotation1 = mn.Quaternion()

    forward_v = forward_v * (mn.Vector3(1.0, 1.0, 1.0) - mn.Vector3.y_axis())
    angle2 = mn.math.angle(forward_v.normalized(), -1 * mn.Vector3.z_axis())
    axis2 = mn.Vector3.y_axis()
    rotation2 = mn.Quaternion.rotation(angle2, axis2)

    return rotation2 * rotation1


class MocapInterface:
    def __init__(
        self,
        sim,
        fps,
        bvh_path=None,
        urdf_path=None,
        rotation_offset=mn.Quaternion(),
        translate_offset=mn.Vector3([0, 0, 0]),
        motion_start=None,
        motion_end=None,
    ) -> None:
        # general interface attrs
        self.bvh_path = bvh_path
        self.urdf_path = urdf_path

        self.sim: Optional[habitat_sim.simulator.Simulator] = sim
        self.draw_fps: float = fps
        self.art_obj_mgr = self.sim.get_articulated_object_manager()
        self.motion: Optional[motion.Motion] = None
        self.motion_stepper = 0
        self.rotation_offset: Optional[mn.Quaternion] = rotation_offset
        self.translate_offset: Optional[mn.Vector3] = translate_offset
        self.motion_start: int = motion_start
        self.motion_end: int = motion_end
        self.is_reversed = False

        self.model: Optional[phy.ManagedArticulatedObject] = None
        self.load_motion()

    def set_transform_offsets(
        self,
        rotate_offset: Optional[mn.Quaternion] = None,
        translate_offset: Optional[mn.Vector3] = None,
    ) -> None:
        """
        This method updates the offset of the model with the positional data passed to it.
        Use this for changing the location and orientation of the model.
        """
        self.rotation_offset = rotate_offset or self.rotation_offset
        self.translate_offset = translate_offset or self.translate_offset

        self.next_pose(repeat=True)

    def load_motion(self) -> None:
        """
        Loads the motion
        """
        # loading text because the setup pauses here during motion load
        logger.info("Loading...")
        self.motion = cut(
            bvh.load(file=self.bvh_path), self.motion_start, self.motion_end
        )
        self.set_transform_offsets(
            rotate_offset=self.rotation_offset, translate_offset=self.translate_offset
        )
        logger.info("Done Loading.")

    # currently the next_pose method is simply called twice in simulating a frame
    def next_pose(self, repeat=False, step_size=None) -> None:
        """
        Set the model state from the next frame in the motion trajectory. `repeat` is
        set to `True` when the user would like to repeat the last frame.
        """
        # precondition
        if not all([self.model, self.motion]):
            return

        # tracks is_reversed and changes the direction of the motion accordingly.
        def sign(i):
            return -1 * i if self.is_reversed else i

        step_size = step_size or int(self.motion.fps / self.draw_fps)

        # repeat
        if not repeat:
            # iterate the frame counter
            self.motion_stepper = (
                self.motion_stepper + sign(step_size)
            ) % self.motion.num_frames()

        (
            new_pose,
            new_root_translate,
            new_root_rotation,
        ) = self.convert_CMUamass_single_pose(
            self.motion.poses[abs(self.motion_stepper)], self.model, raw=True
        )

        self.model.joint_positions = new_pose
        self.model.rotation = new_root_rotation
        self.model.translation = new_root_translate

    def convert_CMUamass_single_pose(
        self, pose, model, raw=False
    ) -> Tuple[List[float], mn.Vector3, mn.Quaternion]:
        """
        This conversion is specific to the datasets from CMU
        """
        new_pose = []

        # Root joint
        root_T = pose.get_transform(0, local=False)

        final_rotation_correction = mn.Quaternion()

        if not raw:
            final_rotation_correction = (
                global_correction_quat(mn.Vector3.z_axis(), mn.Vector3.x_axis())
                * self.rotation_offset
            )

        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )
        root_translation = (
            self.translate_offset
            + final_rotation_correction.transform_vector(root_T[0:3, 3])
        )

        Q, _ = conversions.T2Qp(root_T)

        # Other joints
        for model_link_id in model.get_link_ids():
            joint_type = model.get_link_joint_type(model_link_id)

            if joint_type == phy.JointType.Fixed:
                continue

            joint_name = model.get_link_name(model_link_id)
            pose_joint_index = pose.skel.index_joint[joint_name]

            # When the target joint do not have dof, we simply ignore it

            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if pose_joint_index is None:
                raise KeyError(
                    "Error: pose data does not have a transform for that joint name"
                )
            elif joint_type not in [phy.JointType.Spherical]:
                raise NotImplementedError(
                    f"Error: {joint_type} is not a supported joint type"
                )
            else:
                T = pose.get_transform(pose_joint_index, local=True)
                if joint_type == phy.JointType.Spherical:
                    Q, _ = conversions.T2Qp(T)

            new_pose += list(Q)

        return new_pose, root_translation, root_rotation

    def load_model(self, motion_type=phy.MotionType.KINEMATIC) -> None:
        """
        Loads the model
        """
        # loading text because the setup pauses here during motion load
        logger.info("Loading...")
        self.hide_model()

        # add an ArticulatedObject to the world with a fixed base
        self.model = self.art_obj_mgr.add_articulated_object_from_urdf(
            filepath=self.urdf_path, fixed_base=True
        )
        assert self.model.is_alive

        # set the motion type
        self.model.motion_type = motion_type

        # if DYNAMIC, disable self collisions
        if motion_type == phy.MotionType.DYNAMIC:
            phy.CollisionGroupHelper.set_group_interacts_with(
                phy.CollisionGroups.UserGroup1,
                phy.CollisionGroups.UserGroup1,
                False,
            )
            self.model.override_collision_group(phy.CollisionGroups.UserGroup1)

        self.model.translation = self.translate_offset
        self.next_pose(repeat=True)
        logger.info("Done Loading.")

    def hide_model(self) -> None:
        """
        Removes model from scene.
        """
        if self.model:
            self.art_obj_mgr.remove_object_by_handle(self.model.handle)
            self.model = None
