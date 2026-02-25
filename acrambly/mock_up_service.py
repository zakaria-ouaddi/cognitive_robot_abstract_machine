import geometry_msgs.msg
from perception_interfaces.srv import GetObjPoses
import rclpy.node as node


class GetCubePosesService(node.Node):

    def __init__(self):
        super().__init__('perception')
        self.srv = self.create_service(GetObjPoses, 'cube_poses', self.callback)

    def callback(self, request, response):
        response.pose = detect_pose(request.object)
        return response

def detect_pose(obj):
    print(obj)
    if obj == "blue_box":
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.frame_id = "camera_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = 1.0, 0.0, 0.25
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = 0.0, 0.0, 0.0, 1.0
        print(pose)
        return pose
    if obj == "red_cube":
        return geometry_msgs.msg.PoseStamped(pose=[0, 0, 1], orientation=[0, 0, 0, 1])

import rclpy
rclpy.init()

service = GetCubePosesService()
rclpy.spin(service)
rclpy.shutdown()