from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix as TransformationMatrix
import math

world = World()

# Create an explicit root body like in other examples, so we can attach our base
root = Body(name=PrefixedName(name="root", prefix="world"))

# Geometry for a square base plate and a "camera" box
base_plate_shape = Box(scale=Scale(0.6, 0.6, 0.05), color=Color(0.3, 0.3, 0.3, 1.0))
camera_shape = Box(scale=Scale(0.10, 0.06, 0.05), color=Color(0.1, 0.2, 0.8, 1.0))

base_body = Body(
    name=PrefixedName(name="base", prefix="transform_example"),
    visual=ShapeCollection([base_plate_shape]),
    collision=ShapeCollection([base_plate_shape]),
)


camera_body = Body(
    name=PrefixedName(name="camera", prefix="transform_example"),
    visual=ShapeCollection([camera_shape]),
    collision=ShapeCollection([camera_shape]),
)

camera_body2 = Body(
    name=PrefixedName(name="camera2", prefix="transform_example"),
    visual=ShapeCollection([camera_shape]),
    collision=ShapeCollection([camera_shape]),
)

# Place the base: put it slightly above the ground so the top sits at z≈0.05
world_T_base = TransformationMatrix.from_xyz_rpy(z=0.025, reference_frame=root)

# Place the camera relative to the base: forward along base-y and a bit upward, yawed by +30°
base_T_camera = TransformationMatrix.from_xyz_rpy(
    z=0.025,
    yaw=math.radians(30),
    reference_frame=base_body,
)

with world.modify_world():
    # Connect base to the world root, and camera to the base.
    world_C_base = Connection6DoF.create_with_dofs(parent=root, child=base_body, world=world)
    base_C_camera = Connection6DoF.create_with_dofs(parent=base_body, child=camera_body, world=world)

    world.add_connection(world_C_base)
    world.add_connection(base_C_camera)

# Set origins in a separate modification block so FK is compiled first
with world.modify_world():
    world_C_base.origin = world_T_base
    base_C_camera.origin = base_T_camera



# Visualize
rt = RayTracer(world)
rt.update_scene()
rt.scene.show()
matrix = camera_body.parent_connection.origin @ camera_body.parent_connection.parent.parent_connection.origin

print(matrix)
with world.modify_world():
    with world.modify_world():
        world.get_connection_by_name(PrefixedName("root_T_base", "transform_example")).origin = world_T_base
        world.get_connection_by_name(PrefixedName("base_T_camera", "transform_example")).origin = base_T_camera = TransformationMatrix.from_xyz_rpy(
            x=0.25,
            y=0.25,
            z=0.10,
            yaw=math.radians(30),
            reference_frame=base_body,
)


rt.update_scene()
rt.scene.show()



with world.modify_world():
    world_C_camera = Connection6DoF.create_with_dofs(parent=root, child=camera_body2, world=world)
    world.add_connection(world_C_camera)
    world_C_camera.origin = matrix

rt.update_scene()
rt.scene.show()