"""
Realistic benchmark: Graph of Convex Sets (GCS) free-space calculation on the IAI
Apartment world loaded via the semantic_digital_twin package.

The benchmark measures each phase of the pipeline independently so we can see
exactly where time is spent:

  1. World loading   – URDF parse + pybullet collision setup
  2. Obstacle union  – reduce(or_, obstacle_events) over all ~70 obstacle BBs
  3. Free space      – ~obstacles & search_event  (product-algebra hot path)
  4. Materialise     – BoundingBoxCollection.from_event()
  5. Connectivity    – R-tree intersection graph

Run with:
    python3.12 random_events/benchmarks/benchmark_gcs_apartment.py

Requirements (all installed in-repo):
    semantic_digital_twin, random_events, rtree, trimesh, urdf_parser_py
"""
from __future__ import annotations

import os
import sys
import time
import statistics
from functools import reduce
from operator import or_

# ── path setup ─────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("semantic_digital_twin/src", "krrood/src", "giskardpy/src"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ───────────────────────────────────────────────────────────────────────────

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

_APARTMENT_URDF = os.path.join(
    _ROOT,
    "semantic_digital_twin",
    "resources",
    "urdf",
    "apartment.urdf",
)
_FALLBACK_MESH = os.path.join(os.path.dirname(__file__), "_fallback_cube.obj")

# ── fallback mesh: a minimal unit cube (used when ROS packages are absent) ─
_CUBE_OBJ = """\
# minimal unit cube – placeholder for unresolvable package:// meshes
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""

if not os.path.isfile(_FALLBACK_MESH):
    with open(_FALLBACK_MESH, "w") as _f:
        _f.write(_CUBE_OBJ)


class FallbackPathResolver(PathResolver):
    """
    Resolves any URI that CompositePathResolver can handle normally;
    for all others (e.g. package://iai_apartment/…) returns a tiny
    placeholder OBJ cube so that trimesh can still load *something*.
    """

    _inner = CompositePathResolver()

    def supports(self, uri: str) -> bool:
        return True

    def resolve(self, uri: str) -> str:
        try:
            return self._inner.resolve(uri)
        except Exception:
            return _FALLBACK_MESH


# ── helpers ─────────────────────────────────────────────────────────────────

def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _timeit(label: str, fn, *, n: int = 1):
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    if n > 1:
        stdev = statistics.stdev(times)
        print(f"  {label:<40s}  {_ms(mean):>10s}  ±{_ms(stdev)}")
    else:
        print(f"  {label:<40s}  {_ms(mean):>10s}")
    return result


# ── benchmark ───────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("GCS Free-Space Benchmark  –  IAI Apartment World")
    print("=" * 65)

    # ── Phase 1: World loading ───────────────────────────────────────────
    print("\n[1] World loading")

    def _load_world():
        parser = URDFParser.from_file(
            _APARTMENT_URDF, path_resolver=FallbackPathResolver()
        )
        return parser.parse()

    world = _timeit("URDF parse + pybullet setup", _load_world)

    n_bodies = len(list(world.bodies))
    n_collision = sum(
        1 for b in world.bodies if isinstance(b, Body) and b.has_collision()
    )
    print(f"      bodies={n_bodies}  with-collision={n_collision}")

    # ── Phase 2: Define search space ─────────────────────────────────────
    print("\n[2] Search space")
    # Apartment furnitures_root is at (8.85, 1.75, 0).  Walls span roughly
    # x ∈ [-1, 12],  y ∈ [-3, 5],  z ∈ [0, 3].
    search_space = BoundingBoxCollection(
        shapes=[
            BoundingBox(
                min_x=-1.0,
                min_y=-3.0,
                min_z=0.0,
                max_x=12.0,
                max_y=5.0,
                max_z=3.0,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        reference_frame=world.root,
    )
    search_event = search_space.event
    print(f"  Search box  x[-1,12]  y[-3,5]  z[0,3]")
    print(f"  Search event variables: {list(search_event.variables)}")

    # ── Phase 3: Collect obstacle bounding boxes ─────────────────────────
    print("\n[3] Collect obstacle bounding boxes")

    def _collect_bbs():
        annotation = SemanticEnvironmentAnnotation(root=world.root, _world=world)
        origin = HomogeneousTransformationMatrix(reference_frame=world.root)
        return list(annotation.as_bounding_box_collection_at_origin(origin))

    obstacle_bbs = _timeit("as_bounding_box_collection_at_origin", _collect_bbs)
    print(f"      obstacle bounding boxes: {len(obstacle_bbs)}")

    # ── Phase 4: Build obstacle event (reduce union) ─────────────────────
    print("\n[4] Build obstacle event via reduce(or_, …)  [OLD pipeline]")

    def _build_obstacle_event():
        events = (
            bb.simple_event.as_composite_set() & search_event
            for bb in obstacle_bbs
        )
        events = (e for e in events if not e.is_empty())
        return reduce(or_, events)

    obstacles = _timeit("reduce(or_, obstacle_events)", _build_obstacle_event, n=3)
    n_obs_simple = len(list(obstacles.simple_sets))
    print(f"      obstacle event simple sets: {n_obs_simple}")

    # ── Phase 5: Complement + intersection (free space) ──────────────────
    print("\n[5] Free-space: ~obstacles & search_event  [OLD: complement in ℝ³]")

    def _free_space():
        return ~obstacles & search_event

    free_space = _timeit("~obstacles & search_event", _free_space, n=3)
    n_free_simple = len(list(free_space.simple_sets))
    print(f"      free-space simple sets: {n_free_simple}")

    # ── Phase 4+5 NEW: subtract_disjoint (bounded subtraction) ───────────
    print("\n[4+5 NEW] Free-space via subtract_disjoint  [bounded, no complement]")
    print("          Replaces both phase 4 (reduce union) and phase 5 (complement)")

    def _free_space_subtract():
        fs = search_event
        for bb in obstacle_bbs:
            obstacle = bb.simple_event.as_composite_set() & search_event
            if not obstacle.is_empty():
                fs = fs.subtract_disjoint(obstacle)
        return fs

    free_space_new = _timeit("subtract_disjoint loop (all obstacles)", _free_space_subtract, n=3)
    n_free_new = len(list(free_space_new.simple_sets))
    print(f"      free-space simple sets: {n_free_new}")

    # Verify equivalence: both methods should describe the same set
    diff_check = free_space_new.subtract_disjoint(free_space)
    print(f"      new ⊆ old (extra pieces in new): {diff_check.is_empty()}")

    # ── Phase 6: Materialise into BoundingBoxCollection ──────────────────
    print("\n[6] Materialise free space → BoundingBoxCollection")

    def _materialise():
        return BoundingBoxCollection.from_event(
            reference_frame=world.root, event=free_space
        )

    bbc = _timeit("BoundingBoxCollection.from_event", _materialise, n=3)
    print(f"      free-space bounding boxes: {len(bbc)}")

    # ── Phase 7: Connectivity (R-tree) ────────────────────────────────────
    print("\n[7] Connectivity (R-tree)")

    def _connectivity():
        gcs = GraphOfConvexSets(world=world, search_space=search_space)
        for bb in bbc:
            gcs.add_node(bb)
        gcs.calculate_connectivity(tolerance=0.001)
        return gcs

    gcs = _timeit("calculate_connectivity", _connectivity, n=3)
    print(f"      nodes={len(gcs.graph.nodes())}  edges={len(gcs.graph.edges())}")

    # ── Phase 8: End-to-end (single call) ────────────────────────────────
    print("\n[8] Full end-to-end: GraphOfConvexSets.free_space_from_world")

    def _end_to_end():
        w2 = _load_world()
        ss2 = BoundingBoxCollection(
            shapes=[
                BoundingBox(
                    min_x=-1.0,
                    min_y=-3.0,
                    min_z=0.0,
                    max_x=12.0,
                    max_y=5.0,
                    max_z=3.0,
                    origin=HomogeneousTransformationMatrix(reference_frame=w2.root),
                )
            ],
            reference_frame=w2.root,
        )
        return GraphOfConvexSets.free_space_from_world(w2, ss2)

    full_gcs = _timeit("free_space_from_world (incl. world load)", _end_to_end)
    print(
        f"      nodes={len(full_gcs.graph.nodes())}  "
        f"edges={len(full_gcs.graph.edges())}"
    )

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
