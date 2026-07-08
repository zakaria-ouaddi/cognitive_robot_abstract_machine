from __future__ import annotations

import math
from typing import Dict, Optional, Union, List

from coraplex.datastructures.enums import Grasp, ApproachDirection, VerticalAlignment


class Rotations(Dict[Optional[Union[Grasp, bool]], List[float]]):
    """
    A dictionary that defines standard quaternions for different grasps and orientations.

    All rotation tables are defined in the **robot's coordinate frame**. When composed
    with the robot's live base orientation (``map_R_robot``) in
    :meth:`~coraplex.datastructures.grasp.GraspDescription._world_frame_gripper_rotation`,
    they are promoted to world-frame rotations and then projected into the object frame.

    SIDE_ROTATIONS: Quaternions for lateral approach directions (FRONT, BACK, LEFT, RIGHT)
        expressed in the robot frame. FRONT corresponds to no rotation (identity), meaning
        the gripper approaches from the direction the robot is facing.
    VERTICAL_ROTATIONS: Quaternions for vertical alignment corrections in the robot frame.
        TOP tilts the gripper downward to grasp from above; BOTTOM tilts it upward.
    HORIZONTAL_ROTATIONS: Quaternions for gripper roll corrections in the robot frame.
        Used when the gripper needs to be rotated 90° around its approach axis.
    """

    SIDE_ROTATIONS = {
        ApproachDirection.FRONT: [0, 0, 0, 1],
        ApproachDirection.BACK: [0, 0, 1, 0],
        ApproachDirection.LEFT: [0, 0, -math.sqrt(2) / 2, math.sqrt(2) / 2],
        ApproachDirection.RIGHT: [0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2],
    }

    VERTICAL_ROTATIONS = {
        VerticalAlignment.NoAlignment: [0, 0, 0, 1],
        VerticalAlignment.TOP: [0, math.sqrt(2) / 2, 0, math.sqrt(2) / 2],
        VerticalAlignment.BOTTOM: [0, -math.sqrt(2) / 2, 0, math.sqrt(2) / 2],
    }

    HORIZONTAL_ROTATIONS = {
        False: [0, 0, 0, 1],
        True: [math.sqrt(2) / 2, 0, 0, math.sqrt(2) / 2],
    }
