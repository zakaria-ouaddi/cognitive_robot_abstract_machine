from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from typing_extensions import Optional, Union, List

from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, RotationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Vector3, Quaternion
from semantic_digital_twin.world_description.world_entity import Body, KinematicStructureEntity
from coraplex.datastructures.rotations import Rotations
from coraplex.datastructures.enums import AxisIdentifier, ApproachDirection, VerticalAlignment, Arms
from coraplex.tf_transformations import quaternion_matrix
from coraplex.utils import translate_pose_along_local_axis


@dataclass
class GraspDescription:
    """
    Describes a grasp configuration for a end_effector the description consists of the approach direction (the side from
    which to grasp e.g. FRONT, LEFT, etc and the vertical alignment (TOP, BOTTOM).
    """

    approach_direction: ApproachDirection
    """
    The direction from which the body should be grasped. These are the four directions in the x-y plane (FRONT, BACK, LEFT, RIGHT).
    """

    vertical_alignment: VerticalAlignment
    """
    The alignment of the gripper with the body in the z-axis (TOP, BOTTOM).
    """

    end_effector: EndEffector
    """
    The end_effector that is used to grasp the body.
    """

    rotate_gripper: bool = False
    """
    Rotate the gripper by 90 degrees.
    """

    manipulation_offset: float = 0.05
    """
    The offset between the center of the pose in the grasp sequence
    """

    def pose_sequence(
        self, target_T_grasp_pose: Pose, body: Body = None, reverse: bool = False
    ) -> List[Pose]:
        """
        Calculates the pose sequence to grasp something at the pose. If ``body`` is given,
        its geometry is also taken into account. The pose sequence consists of 3 poses:
        one in front of the body (taking body geometry into account), one at the center
        of the body, and the last one above the body to lift it.

        :param target_T_grasp_pose: The pose of the grasp in the target frame.
        :param body: The body of the grasp.
        :param reverse: If the sequence should be reversed.
        :return: The pose sequence.
        """
        target = target_T_grasp_pose.reference_frame
        world = target._world

        grasp_pose_R_gripper_goal = self.grasp_orientation(target_T_grasp_pose)

        # Compose only the rotational parts so the translation (grasp position) is preserved.
        target_R_gripper_goal = target_T_grasp_pose.to_rotation_matrix() @ grasp_pose_R_gripper_goal.to_rotation_matrix()
        target_T_gripper_goal: Pose = Pose(
            position=target_T_grasp_pose.to_position(),
            orientation=target_R_gripper_goal.to_quaternion(),
            reference_frame=target,
        )

        if body:
            bb_in_frame = body.collision.as_bounding_box_collection_in_frame(
                body
            ).bounding_box()

            approach_axis = np.array(self.approach_direction.axis.value, dtype=bool)

            # Pre-pose calculation
            offset = (
                np.array(bb_in_frame.dimensions)[approach_axis] / 2
                + self.manipulation_offset
            )[0]
        else:
            offset = 0

        target_T_gripper_goal_copy = deepcopy(target_T_gripper_goal)
        pre_pose = translate_pose_along_local_axis(
            target_T_gripper_goal_copy, self.manipulation_axis(), -offset
        )

        target_T_gripper_goal_copy = deepcopy(target_T_gripper_goal)

        # Lift pose: translate along global z-axis, then express in target frame.
        map_T_grasp = world.transform(target_T_grasp_pose.to_homogeneous_matrix(), world.root)
        grasp_T_lift = HomogeneousTransformationMatrix.from_xyz_rpy(z=self.manipulation_offset)
        map_T_lift = (map_T_grasp @ grasp_T_lift).to_position()
        target_P_lift = world.transform(map_T_lift, target)
        lift_pose = Pose(target_P_lift, target_T_gripper_goal.to_quaternion(), reference_frame=target)

        sequence = [pre_pose, target_T_gripper_goal_copy, lift_pose]

        if reverse:
            sequence.reverse()
        return sequence

    def grasp_pose_sequence(self, body: Body):
        """
        Calculates the pose sequence to grasp the body. The sequence is 3 poses, one in front of the body (taking body
        geometry into account), one at the center of the body, and the last one above the body to lift it.

        :param body: The body of the grasp.
        :return: The pose sequence.
        """
        return self.pose_sequence(Pose(reference_frame=body), body)

    def place_pose_sequence(self, pose: Pose) -> List[Pose]:
        """
        Calculates the pose sequence to place a body at the given pose. Assumes that the end_effector is holding a body
        which is being placed.

        :param pose: The pose at which the body in the end_effector should be placed
        :return: The pose sequence.
        """
        body = self.end_effector.tool_frame.child_kinematic_structure_entities[0]
        return self.pose_sequence(pose, body, reverse=True)

    def manipulation_axis(self) -> List[float]:
        """
        The axis of the end_effector that is used for manipulation (i.e. the approach axis).
        This is a physical property of the end-effector (its forward direction), and does
        not depend on the chosen approach direction for a specific grasp.

        :return: The manipulation axis as a list of [x, y, z] values.
        """
        return self.calculate_manipulator_axis(AxisIdentifier.X)

    def lift_axis(self) -> List[float]:
        """
        The axis of the end_effector that is used for lifting.
        This is the world Z axis (gravity) expressed in the gripper's resting frame.

        :return: The lift axis as a list of [x, y, z] values.
        """
        return self.calculate_manipulator_axis(AxisIdentifier.Z)

    def calculate_manipulator_axis(self, axis: AxisIdentifier) -> List[float]:
        """
        Calculates the corresponding axis of the end_effector for a given axis of the world.
        The manipulation and lift axes are physical properties of the gripper hardware and
        are defined intrinsically by the end-effector's mounting correction.

        :param axis: The axis of the world as an AxisIdentifier.
        :return: The corresponding axis of the end_effector as a list of [x, y, z] values.
        """
        world_axis = np.array(axis.value, dtype=float)
        ee_correction = quaternion_matrix(self.end_effector.front_facing_orientation.to_np())[:3, :3]
        # Redone math: A direct projection into the gripper's local mounting frame.
        return (ee_correction.T @ world_axis).round(4).tolist()

    def _world_frame_gripper_rotation(self) -> np.ndarray:
        """
        Computes the full gripper orientation as a 3×3 rotation matrix in the world (map) frame.

        The computation chains the robot's current base orientation in the world with the
        configured approach-direction, vertical-alignment, and roll rotations (all expressed
        in the robot frame), followed by the end-effector's intrinsic correction.

        .. note::
            The SIDE_ROTATIONS, VERTICAL_ROTATIONS, and HORIZONTAL_ROTATIONS tables encode
            rotations in the **robot coordinate frame**. By pre-multiplying with
            ``map_R_robot``, they are promoted to world-frame rotations.

        :return: A 3×3 numpy rotation matrix representing ``map_R_gripper``.
        """
        robot = self.end_effector._robot
        map_R_robot = robot.root.global_pose.to_rotation_matrix().to_np()[:3, :3]

        robot_R_side = quaternion_matrix(Rotations.SIDE_ROTATIONS[self.approach_direction])[:3, :3]
        robot_R_vertical = quaternion_matrix(Rotations.VERTICAL_ROTATIONS[self.vertical_alignment])[:3, :3]
        robot_R_horizontal = quaternion_matrix(Rotations.HORIZONTAL_ROTATIONS[self.rotate_gripper])[:3, :3]
        ee_R_correction = quaternion_matrix(self.end_effector.front_facing_orientation.to_np())[:3, :3]

        return map_R_robot @ robot_R_side @ robot_R_vertical @ robot_R_horizontal @ ee_R_correction

    def grasp_orientation(self, object_pose: Pose) -> Quaternion:
        """
        Computes the gripper orientation expressed in the frame of ``object_pose``.

        Approach directions (FRONT, BACK, LEFT, RIGHT) are defined in the robot's
        coordinate frame.  The world-frame gripper rotation is computed first, then
        projected into the frame of ``object_pose`` via:

        .. code-block:: text

            pose_R_gripper = pose_R_map @ map_R_gripper

        This is the rotation that, when composed with ``object_pose``'s own rotation
        inside :meth:`_pose_sequence`, yields the correct world-frame gripper orientation:

        .. code-block:: text

            target_R_gripper = target_R_pose @ pose_R_gripper
                             = target_R_pose @ pose_R_map @ map_R_gripper
                             = target_R_map @ map_R_gripper

        :param object_pose: Pose whose reference frame is the frame in which the
            returned quaternion is expressed.  For :meth:`grasp_pose` this is a
            world-identity pose in the body frame; for :meth:`_pose_sequence` this
            is ``target_T_grasp_pose`` itself.
        :return: Gripper orientation quaternion expressed in the frame of
            ``object_pose``.
        """
        map_R_gripper = self._world_frame_gripper_rotation()

        # Rotation of object_pose expressed in world frame — this is the rotation
        # of the pose itself, not a separate world-frame lookup.
        pose_R_world = object_pose.to_rotation_matrix().to_np()[:3, :3].T

        pose_R_gripper = pose_R_world @ map_R_gripper
        return RotationMatrix(pose_R_gripper).to_quaternion()

    def edge_offset(self, body: Body) -> float:
        """
        The offset between the center of the body and its edge in the direction of the approach axis.

        :param body: The body to calculate the edge offset for.
        :return: The edge offset.

        """
        rim_direction_index = self.approach_direction.value[0].value.index(1)

        rim_offset = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .dimensions[rim_direction_index]
            / 2
        )
        return rim_offset

    def grasp_pose(self, body: Body, grasp_edge: bool = False) -> Pose:
        """
        The pose for the given end_effector to grasp the body, expressed in the body's frame.

        :param body: The body to grasp.
        :param grasp_edge: Indicates if the pose should be at the edge of the body rather than its centre.
        :return: The grasp pose in the body frame.
        """
        edge_offset = -self.edge_offset(body) if grasp_edge else 0
        body_world_pose = body.global_pose
        orientation = self.grasp_orientation(body_world_pose)
        return Pose(Point3(edge_offset, 0, 0), orientation, reference_frame=body)

    @classmethod
    def calculate_grasp_descriptions(
        cls,
        end_effector: EndEffector,
        pose: Pose,
        grasp_alignment: Optional[PreferredGraspAlignment] = None,
    ) -> List[GraspDescription]:
        """
        This method determines the possible grasp configurations (approach axis and vertical alignment) of the body,
        taking into account the bodies orientation, position, and whether the gripper should be rotated by 90°.

        :param end_effector: The end_effector to use.
        :param grasp_alignment: An optional PreferredGraspAlignment object that specifies preferred grasp axis,
        :param pose: The pose of the object to be grasped.

        :return: A sorted list of GraspDescription instances representing all grasp permutations.
        """
        world = end_effector._world
        map_T_object = world.transform(pose.to_homogeneous_matrix(), world.root).to_pose()

        map_T_robot = end_effector._robot.root.global_pose

        if grasp_alignment:
            side_axis = grasp_alignment.preferred_axis
            vertical = grasp_alignment.with_vertical_alignment
            rotated_gripper = grasp_alignment.with_rotated_gripper
        else:
            side_axis, vertical, rotated_gripper = (
                AxisIdentifier.Undefined,
                False,
                False,
            )

        map_P_object = map_T_object.to_position()
        map_P_robot = map_T_robot.to_position()

        map_V_robot_to_object = map_P_robot - map_P_object


        object_R_map = map_T_object.to_rotation_matrix().inverse()

        object_V_robot = object_R_map @ map_V_robot_to_object

        vector_side = Vector3(object_V_robot.x, object_V_robot.y, np.nan)
        side_faces = GraspDescription.calculate_closest_faces(vector_side, side_axis)

        vector_vertical = Vector3(np.nan, np.nan, object_V_robot.z)
        if vertical:
            vertical_faces = GraspDescription.calculate_closest_faces(vector_vertical)
        else:
            vertical_faces = [VerticalAlignment.NoAlignment]

        grasp_configs = [
            GraspDescription(
                approach_direction=side,
                vertical_alignment=top_face,
                rotate_gripper=rotated_gripper,
                end_effector=end_effector,
            )
            for top_face in vertical_faces
            for side in side_faces
        ]

        return grasp_configs

    @staticmethod
    def calculate_closest_faces(
        pose_to_robot_vector: Vector3,
        specified_grasp_axis: AxisIdentifier = AxisIdentifier.Undefined,
    ) -> Union[
        Tuple[ApproachDirection, ApproachDirection],
        Tuple[VerticalAlignment, VerticalAlignment],
    ]:
        """
        Determines the faces of the object based on the input vector.

        If `specified_grasp_axis` is None, it calculates the primary and secondary faces based on the vector's magnitude
        determining which sides of the object are most aligned with the robot. This will either be the x, y plane for side faces
        or the z axis for top/bottom faces.
        If `specified_grasp_axis` is provided, it only considers the specified axis and calculates the faces aligned
        with that axis.

        :param pose_to_robot_vector: A 3D vector representing one of the robot's axes in the pose's frame, with
                              irrelevant components set to np.nan.
        :param specified_grasp_axis: Specifies a specific axis (e.g., X, Y, Z) to focus on.

        :return: A tuple of two Grasp enums representing the primary and secondary faces.
        """
        all_axes = [AxisIdentifier.X, AxisIdentifier.Y, AxisIdentifier.Z]

        if not specified_grasp_axis == AxisIdentifier.Undefined:
            valid_axes = [specified_grasp_axis]
        else:
            valid_axes = [
                axis
                for axis in all_axes
                if not np.isnan(pose_to_robot_vector.to_list()[axis.value.index(1)])
            ]

        object_to_robot_vector = np.array(pose_to_robot_vector.to_list()) + 1e-9
        sorted_axes = sorted(
            valid_axes,
            key=lambda axis: abs(object_to_robot_vector[axis.value.index(1)]),
            reverse=True,
        )

        primary_axis: AxisIdentifier = sorted_axes[0]
        primary_sign = int(np.sign(object_to_robot_vector[primary_axis.value.index(1)]))

        primary_axis_class = (
            VerticalAlignment if primary_axis == AxisIdentifier.Z else ApproachDirection
        )
        primary_face = primary_axis_class.from_axis_direction(
            primary_axis, primary_sign
        )

        if len(sorted_axes) > 1:
            secondary_axis: AxisIdentifier = sorted_axes[1]
            secondary_sign = int(
                np.sign(object_to_robot_vector[secondary_axis.value.index(1)])
            )
        else:
            secondary_axis: AxisIdentifier = primary_axis
            secondary_sign = -primary_sign

        secondary_axis_class = (
            VerticalAlignment
            if secondary_axis == AxisIdentifier.Z
            else ApproachDirection
        )
        secondary_face = secondary_axis_class.from_axis_direction(
            secondary_axis, secondary_sign
        )

        return primary_face, secondary_face

    def __hash__(self):
        return id(self)

@dataclass
class PreferredGraspAlignment:
    """
    Description of the preferred grasp alignment for an object.
    """

    preferred_axis: Optional[AxisIdentifier]
    """
    The preferred axis, X, Y, or Z, for grasping the object, or None if not specified.
    """

    with_vertical_alignment: bool
    """
    Indicates if the object should be grasped with a vertical alignment.
    """

    with_rotated_gripper: bool
    """
    Indicates if the gripper should be rotated by 90° around X.
    """

@dataclass(eq=False, init=False)
class GraspPose(Pose):
    """
    A pose from which a grasp can be performed along with the respective arm and grasp description.
    """

    arm: Arms = None
    """
    Arm corresponding to the grasp pose.
    """
    grasp_description: GraspDescription = None
    """
    Grasp description corresponding to the grasp pose.
    """
    approach: str = ''
    """
    Classified approach direction for this grasp pose, as returned by the Simox planner.
    One of 'top', 'front', 'back', 'left', 'right', or '' if unknown.
    Used by SimoxPickUpAction to adapt the lift strategy (height and motion type).
    """
    quality: float = 0.0
    """
    Simox wrench-space grasp quality score (0.0–1.0). Higher is better.
    Used to rank candidates within the same approach direction group.
    """

    def __init__(
            self,
            position: Optional[Point3] = None,
            orientation: Optional[Quaternion] = None,
            reference_frame: Optional[KinematicStructureEntity] = None,
            arm: Arms = None,
            grasp_description: GraspDescription = None,
            approach: str = '',
            quality: float = 0.0,
        ):
        super().__init__(position, orientation, reference_frame)
        self.arm = arm
        self.grasp_description = grasp_description
        self.approach = approach
        self.quality = quality

    @classmethod
    def from_pose(
        cls,
        pose: Pose,
        arm: Arms,
        grasp_description: GraspDescription,
        approach: str = '',
        quality: float = 0.0,
    ) -> 'GraspPose':
        return cls(
            position=pose.to_position(),
            orientation=pose.to_quaternion(),
            reference_frame=pose.reference_frame,
            arm=arm,
            grasp_description=grasp_description,
            approach=approach,
            quality=quality,
        )