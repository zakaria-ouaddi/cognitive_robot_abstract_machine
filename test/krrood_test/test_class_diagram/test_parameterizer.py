from dataclasses import dataclass
from enum import Enum
from typing_extensions import Optional

from krrood.class_diagrams.parameterizer import Parameterizer
from random_events.variable import Continuous, Integer, Symbolic, Variable


@dataclass
class Position:
    x: float
    y: float
    z: float

@dataclass
class Orientation:
    x: float
    y: float
    z: float
    w: Optional[float]

@dataclass
class Pose:
    position: Position
    orientation: Orientation

class Element(Enum):
    C = "c"
    H = "h"

@dataclass
class Atom:
    element: Element
    type: int
    charge: float

def test_parameterizer_example_classes():
    param = Parameterizer()

    # Extract variables for each example class
    pos_vars = param(Position)
    ori_vars = param(Orientation)
    pose_vars = param(Pose)
    atom_vars = param(Atom)

    print("\nPosition Variables:", [v.name for v in pos_vars])
    print("Orientation Variables:", [v.name for v in ori_vars])
    print("Pose Variables:", [v.name for v in pose_vars])
    print("Atom Variables:", [v.name for v in atom_vars])

    # Assertions
    assert all(isinstance(v, Continuous) for v in pos_vars), "Position should have only Continuous"
    assert all(isinstance(v, Continuous) for v in ori_vars), "Orientation should have only Continuous"
    assert any(isinstance(v, Symbolic) for v in atom_vars), "Atom should have Symbolic variables"
    assert any(isinstance(v, Integer) for v in atom_vars), "Atom should have Integer variables"
    assert any(isinstance(v, Continuous) for v in atom_vars), "Atom should have Continuous variables"

    # # Create PCS for Atom variables
    variables_for_pcs = [
        Continuous(v.name) if isinstance(v, Integer) else v for v in atom_vars
    ]
    pcs = param.create_fully_factorized_distribution(variables_for_pcs)
    print("PCS variables:", pcs.variables)
    assert set(pcs.variables) == set(variables_for_pcs), "PCS should include all Atom variables"

