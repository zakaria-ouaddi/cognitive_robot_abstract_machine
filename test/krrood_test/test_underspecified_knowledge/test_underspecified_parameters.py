from copy import deepcopy
from datetime import datetime
import numpy as np
import pytest
from enum import Enum

from krrood.entity_query_language.factories import (
    variable,
    underspecified,
    variable_from,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.grasp import GraspDescription
from random_events.interval import singleton, reals
from random_events.set import Set
from random_events.variable import Symbolic, Continuous
from semantic_digital_twin.orm.model import Point3Mapping
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2

from krrood_test.dataset.example_classes import (
    KRROODPosition,
    KRROODPose,
    KRROODOrientation,
    ListOfEnum,
    TestEnum,
    Atom,
    Element,
)
from semantic_digital_twin.spatial_types import Point3


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


def test_enum_domain(mutable_model_world):
    """
    Test that a KRROOD variable with an Enum domain is correctly handled.
    """
    world, robot_view, context = mutable_model_world

    prob_q = underspecified(GraspDescription)(
        approach_direction=variable(
            TestEnum, [TestEnum.OPTION_A, TestEnum.OPTION_B, TestEnum.OPTION_C]
        ),
        vertical_alignment=...,
        manipulator=variable(Manipulator, world.semantic_annotations),
        rotate_gripper=...,
        manipulation_offset=...,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables

    assert len(variables) > 5
    assert len(parameters.truncation_assignments_from_krrood_variables) == 2


def test_assignments_for_conditioning():
    """
    Test that assignments_for_conditioning returns only literal facts.
    """
    prob_q = underspecified(KRROODPosition)(
        x=1.0, y=..., z=variable(float, domain=[2.0, 3.0])
    )
    parameters = UnderspecifiedParameters(prob_q)
    assignments = parameters.conditioning_assignments_from_literal_values

    variables = parameters.variables
    # The variable name for 'x' literal should be 'KRROODPosition.x'
    x_var = variables.get("KRROODPosition.x")

    assert x_var in assignments
    assert assignments[x_var] == 1.0
    assert len(assignments) == 1


def test_construct_instance_from_model_sample_types():
    """
    Test construct_instance_from_model_sample for various types.
    """
    prob_q = underspecified(KRROODPosition)(
        x=..., y=..., z=variable(int, domain=[10, 20])
    )
    parameters = UnderspecifiedParameters(prob_q)
    vars_list = list(parameters.variables.values())

    x_var = parameters.variables["KRROODPosition.x"]
    y_var = parameters.variables["KRROODPosition.y"]
    z_var = parameters.variables["KRROODPosition.z"]

    sample_data = {x_var: 1.5, y_var: 2.5, z_var: 10}
    sample_array = np.array([sample_data[v] for v in vars_list], dtype=object)

    instance = parameters.construct_instance_from_model_sample(vars_list, sample_array)
    assert isinstance(instance, KRROODPosition)
    assert instance.x == 1.5
    assert instance.y == 2.5
    assert instance.z == 10


def test_union_types_easy():
    prob_q = underspecified(KRROODPosition)(x=..., y=..., z=...)
    prob_q = prob_q.where(
        prob_q.variable.x < 5.0,
    )
    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["KRROODPosition.x"].domain == reals()


def test_union_types():
    prob_q = underspecified(Point3)(x=..., y=..., z=variable(int, domain=[10, 20]))
    prob_q = prob_q.where(
        prob_q.variable.x < 5.0,
    )

    parameters = UnderspecifiedParameters(prob_q)
    variables = parameters.variables
    assert variables["Point3.x"].domain == reals()
