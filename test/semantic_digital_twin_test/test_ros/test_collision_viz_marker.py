from dataclasses import dataclass, field
from time import sleep

from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.adapters.ros.visualization.collision_viz_marker import (
    CollisionVizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot


@dataclass
class MarkerArrayRecorder:
    last_msg: MarkerArray = field(init=False, default=None)

    def __call__(self, msg: MarkerArray):
        self.last_msg = msg


def _avoid_robot_environment_collisions(world):
    """
    Adds a rule so the robot avoids both environment bodies and rebuilds the matrix.
    """
    robot = world.get_semantic_annotations_by_type(MinimalRobot)[0]
    environment = world.get_kinematic_structure_entity_by_name("environment")
    environment2 = world.get_kinematic_structure_entity_by_name("environment2")
    collision_manager = world.collision_manager
    collision_manager.temporary_rules.append(
        AvoidCollisionBetweenGroups(
            buffer_zone_distance=10,
            violated_distance=0.0,
            body_group_a=[robot.root],
            body_group_b=[environment, environment2],
        )
    )
    collision_manager.update_collision_matrix()
    return collision_manager


def _wait_for_message(recorder):
    for _ in range(30):
        if recorder.last_msg is not None:
            break
        sleep(0.1)
    else:
        assert False, "Callback timed out"


def _subscribe(node, publisher):
    recorder = MarkerArrayRecorder()
    node.create_subscription(
        msg_type=MarkerArray,
        topic=publisher.topic_name,
        callback=recorder,
        qos_profile=publisher.qos_profile,
    )
    return recorder


def test_publishes_line_list_on_collision_check(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    publisher = CollisionVizMarkerPublisher(node=rclpy_node)
    collision_manager.add_collision_consumer(publisher)
    recorder = _subscribe(rclpy_node, publisher)

    collisions = collision_manager.compute_collisions()
    assert collisions.any()
    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    marker = recorder.last_msg.markers[0]
    assert marker.type == Marker.LINE_LIST
    assert marker.header.frame_id == str(cylinder_bot_world.root.name)
    assert len(marker.points) == 2 * len(collisions.contacts)
    assert len(marker.colors) == len(marker.points)


def test_color_depends_on_distance(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    # Threshold above every contact distance, so all contacts must be red.
    publisher = CollisionVizMarkerPublisher(
        node=rclpy_node, collision_distance_threshold=100.0
    )
    collision_manager.add_collision_consumer(publisher)
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    colors = recorder.last_msg.markers[0].colors
    assert colors
    for color in colors:
        assert color.r == 1.0
        assert color.g == 0.0


def test_color_green_when_above_threshold(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    # Threshold below every contact distance, so all contacts must be green.
    publisher = CollisionVizMarkerPublisher(
        node=rclpy_node, collision_distance_threshold=-100.0
    )
    collision_manager.add_collision_consumer(publisher)
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()

    _wait_for_message(recorder)
    colors = recorder.last_msg.markers[0].colors
    assert colors
    for color in colors:
        assert color.r == 0.0
        assert color.g == 1.0


def test_throttle_publishes_every_nth_check(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    publisher = CollisionVizMarkerPublisher(node=rclpy_node, throttle=2)
    collision_manager.add_collision_consumer(publisher)
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()
    sleep(0.3)
    assert recorder.last_msg is None

    collision_manager.compute_collisions()
    _wait_for_message(recorder)


def test_with_collision_visualization_wires_consumer(rclpy_node, cylinder_bot_world):
    collision_manager = _avoid_robot_environment_collisions(cylinder_bot_world)
    viz = VizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)
    publisher = viz.with_collision_visualization()

    assert publisher in collision_manager.collision_consumers
    recorder = _subscribe(rclpy_node, publisher)

    collision_manager.compute_collisions()
    _wait_for_message(recorder)
    assert recorder.last_msg.markers[0].type == Marker.LINE_LIST
