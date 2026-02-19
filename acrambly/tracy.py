import time

import rclpy

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import (
    ParkArmsActionDescription,
    PickUpActionDescription,
    PlaceActionDescription,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
# import pycram.robot_descriptions.tracy_states
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix as tm
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

sw = World()
# root = Body(name= PrefixedName('root', "world"))
body_box1 = Body(
    name=PrefixedName("box1", "PhysicalObject"),
)
body_box2 = Body(
    name=PrefixedName("box2", "PhysicalObject"),
)
body_box3 = Body(
    name=PrefixedName("box3", "PhysicalObject"),
)
box1_start = [0.75, 0.00, 0.9, 0, 0, 0]
box2_start = [0.75, 0.50, 0.9, 0, 0, 0]
box3_start = [0.75, 0.25, 0.9, 0, 0, 0]

box1_target = [0.75, 0.50, 0.07, 0, 0, 0]
box2_target = [0.75, 0.50, 0.12, 0, 0, 0]
box3_target = [0.75, 0.50, 0.19, 0, 0, 0]

box1 = Box(tm(reference_frame=body_box1),scale=Scale(0.05,0.05,0.05), color=Color(1,0,0,1))
box2 = Box(tm(reference_frame=body_box2),scale=Scale(0.05,0.05,0.05), color=Color(0,1,0,1))
box3 = Box(tm(reference_frame=body_box3),scale=Scale(0.05,0.05,0.05), color=Color(0,0,1,1))


body_box1.collision = body_box1.visual = ShapeCollection([box1], body_box1)
body_box2.collision = body_box2.visual = ShapeCollection([box2], body_box2)
body_box3.collision = body_box3.visual = ShapeCollection([box3], body_box3)

tracy_world = URDFParser.from_file("../semantic_digital_twin/resources/urdf/tracy.urdf").parse()
#tracy_world = URDFParser.from_file(os.path.join(os.path.dirname(__file__), "..", "resources", "robots", "tracy.urdf")).parse()
from semantic_digital_twin.robots.tracy import Tracy
robot_view = Tracy.from_world(tracy_world)
root = tracy_world.root

print(root)
print(tracy_world.get_kinematic_structure_entity_by_name(PrefixedName("camera_link", "tracy")))

#add boxes to world
with tracy_world.modify_world():
    c_root_box1 = Connection6DoF.create_with_dofs(world=tracy_world, parent=root, child=body_box1)
    tracy_world.add_connection(c_root_box1)

    c_root_box2 = Connection6DoF.create_with_dofs(world=tracy_world, parent=root, child=body_box2)
    tracy_world.add_connection(c_root_box2)

    c_root_box3 = Connection6DoF.create_with_dofs(world=tracy_world, parent=root, child=body_box3)
    tracy_world.add_connection(c_root_box3)
    c_root_box1.origin = tm.from_xyz_rpy(*box1_start, reference_frame=root, child_frame=body_box1)
    c_root_box2.origin = tm.from_xyz_rpy(*box2_start, reference_frame=root)
    c_root_box3.origin = tm.from_xyz_rpy(*box3_start, reference_frame=root)

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node("semantic_world")

viz = VizMarkerPublisher(world=tracy_world, node=node)



print("testing")
#time.sleep(5)
# print("ik test")
# ik1 = tracy_world.compute_inverse_kinematics(
#     robot_view.root,
#     robot_view.left_arm.tip,
#     target=tracy_world.transform(tm.from_xyz_rpy(*box1_target, reference_frame=root), target_frame=root),
#     max_iterations=1000
# )
# with tracy_world.modify_world():
#     for joint, position in ik1.items():
#         tracy_world.state[joint.name].position = position
# print(ik1)

# time.sleep(5)
print("plan test")
time.sleep(5)
# park_description = ParkArmsActionDescription([Arms.BOTH])
# pick_description = PickUpActionDescription(
#     object_designator=body_box1,
#     grasp_description=GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP),
#     arm=Arms.LEFT
# )

print(PoseStamped.from_list(frame=root, position=box1_target[0:3]))
#import pycram.robot_descriptions.tracy_states
rt = RayTracer(tracy_world)
rt.update_scene()
rt.scene.show()

with simulated_robot:
    ctx = Context(world=tracy_world, robot=robot_view)
    SequentialPlan(
        ctx,
        ParkArmsActionDescription([Arms.BOTH]),
        PickUpActionDescription(
            object_designator=body_box1,
            grasp_description=GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, robot_view.left_arm.manipulator),
            arm=Arms.LEFT,
        ),
        PlaceActionDescription(
            object_designator=body_box1,
            target_location=PoseStamped.from_list(frame=robot_view.root, position=box1_target[0:3], orientation=[1,0,0,0]),
            arm=Arms.LEFT,
        ),
        ParkArmsActionDescription([Arms.BOTH]),
        PickUpActionDescription(
            object_designator=body_box3,
            grasp_description=GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, robot_view.left_arm.manipulator),
            arm=Arms.LEFT,
        ),
        PlaceActionDescription(
            object_designator=body_box3,
            target_location=PoseStamped.from_list(frame=robot_view.root, position=box2_target[0:3], orientation=[1,0,0,0]),
            arm=Arms.LEFT,
        ),
        ParkArmsActionDescription([Arms.BOTH]),
    ).perform()

print("done")
node.destroy_node()

rt.update_scene()
rt.scene.show()

try:
    rclpy.shutdown()
except Exception:
    pass