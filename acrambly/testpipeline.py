import rclpy
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

from semantic_digital_twin.world_description.connections import Connection6DoF

from acrambly.pipeline import PerceptionClientSingle
from semantic_digital_twin.adapters.urdf import URDFParser

from semantic_digital_twin.world_description.shape_collection import ShapeCollection

from semantic_digital_twin.world_description.geometry import Box, Scale, Color

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from semantic_digital_twin.world_description.world_entity import Body

from semantic_digital_twin.world import World

from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix as tm

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node("semantic_world")

sw = World()
# root = Body(name= PrefixedName('root', "world"))
body_box1 = Body(
    name=PrefixedName("blue_box", "PhysicalObject"),
)
body_box2 = Body(
    name=PrefixedName("red_box", "PhysicalObject"),
)
body_box3 = Body(
    name=PrefixedName("yellow_box", "PhysicalObject"),
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
rt = RayTracer(tracy_world)
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

perception = PerceptionClientSingle(tracy_world, node)
rt.update_scene()
rt.scene.show()
perception.request("blue_box")
print("Done")

rt.update_scene()
rt.scene.show()