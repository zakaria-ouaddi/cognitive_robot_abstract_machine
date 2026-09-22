"""
pr2_simox_demo/demo.py
=======================
End-to-end demonstration: Simox physics-based grasp planner integrated with
CoraPlex and Giskard.

Pipeline:
  1. Set up the CoraPlex digital twin world (PR2 robot + apartment environment)
  2. Position breakfast_cereal on the table in front of the PR2 right arm
  3. Initialize ROS 2 node, TF broadcaster, and RViz marker publisher
  4. Execute ParkArmsAction (park both arms) and MoveTorsoAction (TorsoState.HIGH)
  5. Execute SimoxPickUpAction:
     a. Queries Simox /plan_grasp ROS 2 service for physics-based grasp candidates
     b. Iterates best-quality-first
     c. Plans and reaches the grasp pose via Giskard
     d. Closes the PR2 gripper and attaches object to gripper link
     e. Lifts object by 0.15 m
  6. Visualise state in RViz2

Prerequisites (run BEFORE this script):
  Terminal 1 - Simox service:
    bash scripts/launch_simox_service.sh

  Terminal 2 (optional) - RViz2:
    rviz2
    Add:
      - TF (Fixed Frame: world)
      - MarkerArray (topic: /semworld/viz_marker, Durability: TRANSIENT_LOCAL)

Usage:
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  PYTHONPATH=cognitive_robot_abstract_machine/coraplex/src:$PYTHONPATH \\
      ~/.virtualenvs/cram-env/bin/python cognitive_robot_abstract_machine/coraplex/demos/pr2_simox_demo/demo.py
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.motion_executor import simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.pick_up import SimoxPickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.testing import setup_world
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('pr2_simox_demo')

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[4]
if not (REPO_ROOT / 'grasp_test_files').exists():
    REPO_ROOT = Path('/home/zakaria/grasp_planner')
OBJECTS_DIR = REPO_ROOT / 'grasp_test_files' / 'resources' / 'objects'
ROBOT_XML = str(REPO_ROOT / 'grasp_test_files' / 'resources' / 'robots' / 'pr2.xml')

# ─── Main Demo Routine ────────────────────────────────────────────────────────


def main():
    logger.info("══════════════════════════════════════════════════════")
    logger.info(" PR2 Simox Grasp Planner + CoraPlex + Giskard Demo    ")
    logger.info("══════════════════════════════════════════════════════")

    # 1. World Setup
    logger.info("Step 1: Setting up CoraPlex world...")
    world = setup_world()

    # Parse CLI arguments early
    import argparse
    parser = argparse.ArgumentParser(description="PR2 Simox Grasp Planner Demo")
    parser.add_argument(
        '--object', type=str, default='milk',
        choices=['milk', 'cereal', 'breakfast_cereal'],
        help="Target object to pick up (default: 'milk'). Choices: 'milk', 'cereal'.",
    )
    parser.add_argument(
        '--spin', action='store_true',
        help="Keep node spinning indefinitely for RViz inspection (Ctrl+C to exit)",
    )
    parser.add_argument(
        '--approach', type=str, default='front',
        choices=['right', 'left', 'front', 'back', 'top', 'any'],
        help="Preferred grasp approach direction (default: 'front'). Use 'any' for purely quality-driven selection.",
    )
    parser.add_argument(
        '--yaw', '--cereal-yaw', type=float, default=0.0, dest='object_yaw',
        help="Object rotation yaw in degrees around Z. Default: 0.0",
    )
    parser.add_argument(
        '--arm', type=str, default='right',
        choices=['right', 'left'],
        help="Arm to use for picking (default: 'right').",
    )
    parser.add_argument(
        '--quality-threshold', type=float, default=0.05,
        help="Minimum grasp wrench quality threshold (default: 0.05).",
    )
    parser.add_argument(
        '--num-grasps', type=int, default=50,
        help="Number of grasp candidates to request from Simox (default: 50).",
    )
    args, _ = parser.parse_known_args()

    # Determine target object and secondary object
    is_milk = (args.object == 'milk')
    target_name = 'milk.stl' if is_milk else 'breakfast_cereal.stl'
    other_name = 'breakfast_cereal.stl' if is_milk else 'milk.stl'

    target_obj = world.get_body_by_name(target_name)
    other_obj = world.get_body_by_name(other_name)

    import math
    yaw_rad = math.radians(args.object_yaw)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)

    # Position target object on counter in front of the selected arm
    # PR2 is located at [1.5, 2.5, 0.0] facing +X
    target_y = 2.35 if args.arm == 'right' else 2.65
    target_pose = Pose.from_xyz_quaternion(
        pos_x=2.15, pos_y=target_y, pos_z=1.05,
        quat_x=0.0, quat_y=0.0, quat_z=float(qz), quat_w=float(qw),
        reference_frame=world.root,
    )
    # Move other object safely aside so it doesn't obstruct the pick
    other_pose = Pose.from_xyz_quaternion(
        pos_x=2.37, pos_y=1.80, pos_z=1.05,
        quat_x=0.0, quat_y=0.0, quat_z=0.0, quat_w=1.0,
        reference_frame=world.root,
    )
    with world.modify_world():
        target_obj.parent_connection.origin = target_pose.to_homogeneous_matrix()
        other_obj.parent_connection.origin = other_pose.to_homogeneous_matrix()

    logger.info(
        "Target '%s' placed at world: x=%.2f y=%.2f z=%.2f (yaw=%.1f°, arm=%s)",
        target_obj.name, float(target_obj.global_pose.x), float(target_obj.global_pose.y), float(target_obj.global_pose.z),
        args.object_yaw, args.arm,
    )

    # 2. ROS 2 Node & RViz Visualization
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node('pr2_simox_demo')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    viz = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    logger.info("RViz visualization publishers active (/semworld/viz_marker, TF)")
    logger.info("RViz Fixed Frame: 'world' (or 'apartment/apartment_root')")

    # 3. Robot & Context
    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, ros_node=node)

    # 4. Demo Execution
    arm_enum = Arms.RIGHT if args.arm == 'right' else Arms.LEFT
    eef_name = 'r_gripper' if args.arm == 'right' else 'l_gripper'
    chain_name = 'RightArm' if args.arm == 'right' else 'LeftArm'

    preferred = None if args.approach.lower() == 'any' else [args.approach.lower()]
    logger.info("Starting demonstration in simulation mode (preferred approach: %s, arm: %s)...", preferred, args.arm)
    with simulated_robot:
        # Step 4a: Park arms & raise torso
        logger.info("── Action: Park arms + raise torso ──")
        execute_single(ParkArmsAction(Arms.BOTH), context=context).plan.perform()
        execute_single(MoveTorsoAction(TorsoState.HIGH), context=context).plan.perform()
        time.sleep(0.5)

        # Step 4b: SimoxPickUpAction
        logger.info("── Action: SimoxPickUpAction for '%s' (preferred=%s) ──", target_obj.name, preferred)
        pickup_action = SimoxPickUpAction(
            object_designator=target_obj,
            arm=arm_enum,
            end_effector_name=eef_name,
            kinematic_chain_name=chain_name,
            preferred_approaches=preferred,
            robot_xml=ROBOT_XML,
            lift_height=0.15,
            num_grasps_to_plan=args.num_grasps,
            quality_threshold=args.quality_threshold,
        )
        execute_single(pickup_action, context=context).plan.perform()

    print("\n══════════════════════════════════════════════════════", flush=True)
    print(f" Demo complete — {target_obj.name} successfully picked up!", flush=True)
    print(f" Final {target_obj.name} pose: x={float(target_obj.global_pose.x):.3f} y={float(target_obj.global_pose.y):.3f} z={float(target_obj.global_pose.z):.3f}", flush=True)
    print("══════════════════════════════════════════════════════\n", flush=True)

    if args.spin:
        print("[demo] Keeping node alive for RViz inspection. Press Ctrl+C to exit.", flush=True)
        try:
            while rclpy.ok():
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[demo] Shutting down pr2_simox_demo...", flush=True)
    else:
        print("[demo] Sleeping 3 seconds to ensure RViz markers are published, then exiting...", flush=True)
        time.sleep(3.0)

    try:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
