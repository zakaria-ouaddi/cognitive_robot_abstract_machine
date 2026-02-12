import geometry_msgs.msg
import rclpy
from example_interfaces.srv import Trigger
from perception_interfaces.srv import GetCubePoses, GetObjPoses

from pycram.datastructures.pose import PoseStamped
from pycram.ros import create_subscriber
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix as TransformationMatrix
from semantic_digital_twin.world import World

prefix = "PhysicalObject"
transform = None


class PerceptionClientSingle():
    def __init__(self, world, node):
        self.world = world
        self.node = node
        self.client = self.node.create_client(GetObjPoses, "cube_poses")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')
        self.req = GetObjPoses.Request()
        self.transform = None

    def update_obj_positions(self, poses, world: World):
        for pose in poses:
            obj_name = pose[0]
            obj_pose = geometry_msgs.msg.PoseStamped()
            obj_trans = TransformationMatrix.from_xyz_quaternion(obj_pose.pose.position.x, obj_pose.pose.position.y,
                                                                 obj_pose.pose.position.z, obj_pose.pose.orientation.x,
                                                                 obj_pose.pose.orientation.y, obj_pose.pose.orientation.z,
                                                                 obj_pose.pose.orientation.w,
                                                                 world.get_kinematic_structure_entity_by_name(
                                                                     PrefixedName("camera_link", "tracy")))
            if not self.transform:
                body = world.get_kinematic_structure_entity_by_name(PrefixedName("camera_link", "tracy"))
                self.transform = body.parent_connection.origin
                body = body.parent_connection.parent
                while body.name != PrefixedName(name='map', prefix='tracy'):
                    print(self.transform)
                    print(body.parent_connection.origin)
                    self.transform = self.transform @ body.parent_connection.origin
                    print(self.transform)
                    body = body.parent_connection.parent

            with world.modify_world():
                world.get_connection_by_name(PrefixedName("map_T_" + obj_name, prefix)).origin = obj_trans @ self.transform

    def request(self, obj):
        self.req.object = obj
        future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        poses = [(obj, response.pose)]
        self.update_obj_positions(poses, self.world)
