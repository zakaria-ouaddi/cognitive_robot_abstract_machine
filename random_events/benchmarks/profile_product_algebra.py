"""
Profiling script: product algebra in 3D with make_disjoint / simplify bottleneck.

Creates N overlapping boxes in 3D, assembles them into a single Event, then
profiles make_disjoint() and simplify() in detail.

Run with:
    python3.12 benchmarks/profile_product_algebra.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import statistics
import timeit

from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous

X = Continuous("x")
Y = Continuous("y")
Z = Continuous("z")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_3d_boxes(n: int, overlap: float = 0.5) -> list[SimpleEvent]:
    """
    Build n boxes in 3D.  Adjacent boxes overlap by `overlap` units so that
    make_disjoint has real work to do.
    """
    boxes = []
    step = 1.0
    size = step + overlap
    for i in range(n):
        lo = float(i) * step
        hi = lo + size
        boxes.append(SimpleEvent.from_data({
            X: closed(lo, hi),
            Y: closed(lo, hi),
            Z: closed(lo, hi),
        }))
    return boxes


def make_event(boxes: list[SimpleEvent]) -> Event:
    return Event.from_simple_sets(*boxes)


# ---------------------------------------------------------------------------
# Wall-time measurements
# ---------------------------------------------------------------------------

def measure(label: str, fn, repeats: int = 5, setup=None):
    """Run fn() `repeats` times, report mean ± stdev in ms."""
    times = []
    for _ in range(repeats):
        if setup:
            obj = setup()
            t = timeit.timeit(lambda: fn(obj), number=1)
        else:
            t = timeit.timeit(fn, number=1)
        times.append(t * 1000)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"  {label:<40} {mean:>9.2f} ms  ±{stdev:.2f}")
    return mean


# ---------------------------------------------------------------------------
# cProfile helpers
# ---------------------------------------------------------------------------

def profile_call(fn, label: str, top_n: int = 20):
    pr = cProfile.Profile()
    pr.enable()
    result = fn()
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(top_n)

    print(f"\n{'─' * 70}")
    print(f"  cProfile: {label}")
    print(f"{'─' * 70}")
    print(buf.getvalue())
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sizes = [5, 10, 20, 30, 50]

    # ── Wall-time table ──────────────────────────────────────────────────────
    print("=" * 70)
    print("Wall-time breakdown — 3D overlapping boxes")
    print("=" * 70)

    for n in sizes:
        boxes = make_3d_boxes(n)
        event = make_event(boxes)

        print(f"\n  N = {n} boxes")
        measure("Event.from_simple_sets()",
                lambda b=boxes: make_event(b))
        measure("event.make_disjoint()",
                lambda e=event: e.make_disjoint())
        measure("event.simplify()",
                lambda e=event: e.simplify())
        measure("event | event  (union, triggers make_disjoint)",
                lambda e=event: e.union_with(e))

    # ── Scaling table: make_disjoint ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("make_disjoint() scaling with N overlapping 3D boxes")
    print("=" * 70)
    print(f"  {'N':>6}  {'ms':>10}  {'growth':>8}")
    prev = None
    for n in [2, 5, 10, 15, 20, 30, 40, 50]:
        boxes = make_3d_boxes(n)
        event = make_event(boxes)
        times = [timeit.timeit(lambda e=event: e.make_disjoint(), number=1) * 1000
                 for _ in range(5)]
        mean = statistics.mean(times)
        growth = f"{mean / prev:.2f}x" if prev else "—"
        print(f"  {n:>6}  {mean:>10.2f}  {growth:>8}")
        prev = mean

    # ── Scaling table: simplify ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("simplify() scaling with N overlapping 3D boxes")
    print("=" * 70)
    print(f"  {'N':>6}  {'ms':>10}  {'growth':>8}")
    prev = None
    for n in [2, 5, 10, 15, 20, 30]:
        boxes = make_3d_boxes(n)
        event = make_event(boxes)
        times = [timeit.timeit(lambda e=event: e.simplify(), number=1) * 1000
                 for _ in range(5)]
        mean = statistics.mean(times)
        growth = f"{mean / prev:.2f}x" if prev else "—"
        print(f"  {n:>6}  {mean:>10.2f}  {growth:>8}")
        prev = mean

    # ── cProfile: make_disjoint on 20 boxes ──────────────────────────────────
    n_profile = 20
    boxes_p = make_3d_boxes(n_profile)
    event_p = make_event(boxes_p)

    profile_call(
        lambda: event_p.make_disjoint(),
        f"make_disjoint() on {n_profile} overlapping 3D boxes",
        top_n=25,
    )

    profile_call(
        lambda: event_p.simplify(),
        f"simplify() on {n_profile} overlapping 3D boxes",
        top_n=25,
    )

    # ── Break down from_simple_sets overhead ─────────────────────────────────
    profile_call(
        lambda: make_event(make_3d_boxes(20)),
        "Event.from_simple_sets() on 20 overlapping 3D boxes",
        top_n=20,
    )


if __name__ == "__main__":
    main()
