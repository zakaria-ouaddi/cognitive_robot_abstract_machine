import os
from copy import deepcopy

import numpy as np
import pytest
import rclpy

from pycram.datastructures.dataclasses import Context

from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Scale, Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session")
def viz_marker_publisher():
    rclpy.init()
    node = rclpy.create_node("test_viz_marker_publisher")
    VizMarkerPublisher(world, node)  # Initialize the publisher
    yield
    rclpy.shutdown()


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


@pytest.fixture(scope="function")
def immutable_model_world(pr2_apartment_world):
    world = pr2_apartment_world
    pr2 = PR2.from_world(world)
    state = deepcopy(world.state.data)
    yield world, pr2, Context(world, pr2)
    world.state.data = state


@pytest.fixture
def immutable_simple_pr2_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup
    state = deepcopy(world.state.data)
    yield world, robot_view, context
    world.state.data = state


@pytest.fixture
def mutable_simple_pr2_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup
    copy_world = deepcopy(world)
    robot_view = PR2.from_world(copy_world)
    return world, robot_view, Context(copy_world, robot_view)
