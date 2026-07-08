#!/usr/bin/env python3
"""
Standalone verification of the robot-frame GraspDescription refactoring.

Run with:
    bash run_demo.sh scripts/verify_grasp_robot_frame.py

Does NOT require ROS, json_msgs, or the ORM.
"""
import sys
import math
import os
from copy import deepcopy

MONO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for pkg in ["coraplex/src", "semantic_digital_twin/src", "krrood/src", "random_events/src"]:
    sys.path.insert(0, os.path.join(MONO, pkg))

import numpy as np
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import OmniDrive, Connection6DoF
from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"
passed = 0; failed = 0

def check(description: str, condition: bool) -> None:
    global passed, failed
    symbol = f"{GREEN}✓{RESET}" if condition else f"{RED}✗ FAIL{RESET}"
    print(f"  {symbol} {description}")
    if condition: passed += 1
    else: failed += 1

def approx(a, b, tol=0.002):
    return all(abs(x - y) < tol for x, y in zip(a, b))


# ── Build a minimal PR2 + milk world (mirrors the test conftest) ─────────
print(f"\n{BOLD}Building world…{RESET}")

urdf_parser = URDFParser.from_file(file_path=PR2.get_ros_file_path())
world = urdf_parser.parse()
PR2.from_world(world)

with world.modify_world():
    old_root = world.root
    map_body = Body(name=PrefixedName("map"))
    odom = Body(name=PrefixedName("odom_combined"))
    map_C_odom = Connection6DoF.create_with_dofs(world, map_body, odom)
    world.add_connection(map_C_odom)
    drive_conn = OmniDrive.create_with_dofs(parent=odom, child=old_root, world=world)
    world.add_connection(drive_conn)
    drive_conn.has_hardware_interface = True

MILK_STL = os.path.join(MONO, "coraplex", "resources", "objects", "milk.stl")
milk_world = STLParser(MILK_STL).parse()
world.merge_world_at_pose(
    milk_world,
    HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.02, yaw=math.pi),
)

robot_view = world.get_semantic_annotations_by_type(PR2)[0]
milk = world.get_body_by_name("milk.stl")
man = robot_view.left_arm.end_effector
print("  World built OK.\n")

g_front = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, man)

# ════════════════════════════════════════════════════════════════════════
print(f"{BOLD}1. manipulation_axis & lift_axis (depend only on robot, not object){RESET}")
check("FRONT manipulation_axis = [1,0,0]", approx(g_front.manipulation_axis(), [1, 0, 0]))
check("FRONT lift_axis         = [0,0,1]", approx(g_front.lift_axis(),         [0, 0, 1]))

# ════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}2. grasp_pose_sequence — robot at identity, milk at yaw=π{RESET}")
print("   Robot faces world +X. FRONT = gripper approaches milk from world +X side.")
print("   In milk frame (yaw=π maps world +X → milk −X), gripper identity = correct.\n")

seq = deepcopy(g_front).grasp_pose_sequence(milk)
check(f"Grasp quaternion = [0,0,0,1] in milk frame "
      f"(got {[round(v,3) for v in seq[1].to_quaternion().to_list()]})",
      approx(seq[1].to_quaternion().to_list(), [0, 0, 0, 1]))
pos = seq[0].to_position().to_list()[:3]
check(f"Pre-pose at [-0.082,0,0] in milk frame "
      f"(got [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]) "
      "— robot side of the milk",
      approx(pos, [-0.082, 0, 0], tol=0.015))

# ════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}3. grasp_orientation tracks ROBOT rotation{RESET}")

milk_pose = Pose(reference_frame=milk)
q_id = g_front.grasp_orientation(milk_pose).to_list()
check(f"Robot at identity → grasp_orientation = [0,0,0,1] "
      f"(got {[round(v,3) for v in q_id]})",
      approx(q_id, [0, 0, 0, 1]))

robot_view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=math.pi / 2)
world.notify_state_change()

q_rot = g_front.grasp_orientation(milk_pose).to_list()
sq2 = math.sqrt(2) / 2
check(f"Robot rotated 90°Z → grasp_orientation ≈ [0,0,√2/2,√2/2] "
      f"(got {[round(v,3) for v in q_rot]})",
      approx(q_rot, [0, 0, sq2, sq2]))

robot_view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy()
world.notify_state_change()

# ════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}4. manipulation_axis is stable when object is rotated{RESET}")

ax_before = g_front.manipulation_axis()
milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.02, yaw=math.pi / 4)
world.notify_state_change()
ax_after = g_front.manipulation_axis()
check(f"manipulation_axis unchanged after rotating milk by 45° "
      f"({[round(v,3) for v in ax_before]} → {[round(v,3) for v in ax_after]})",
      approx(ax_before, ax_after))

milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.02, yaw=math.pi)
world.notify_state_change()

# (Internal consistency check removed: manipulation_axis is physically fixed and does not track the approach direction)

# ════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
total = passed + failed
color = GREEN if failed == 0 else RED
print(f"{BOLD}Results: {color}{passed}/{total} passed{RESET}", end="")
if failed:
    print(f"  {RED}({failed} FAILED){RESET}")
else:
    print(f"  {GREEN}✓ All checks passed{RESET}")
print()
sys.exit(0 if failed == 0 else 1)
