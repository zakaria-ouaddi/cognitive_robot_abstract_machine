"""
Benchmarks for random_events performance improvements.

Compares the old (pre-optimisation) implementations against the new ones
for bounding_box() and fill_missing_variables() across a range of input sizes.

Run with:
    python3.12 benchmarks/benchmark_performance.py
"""

from __future__ import annotations

import statistics
import timeit
from typing import List

from sortedcontainers import SortedSet

from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous


# ---------------------------------------------------------------------------
# Old implementations (preserved verbatim from before the optimisation)
# ---------------------------------------------------------------------------

def bounding_box_old(event: Event) -> SimpleEvent:
    """Original bounding_box: redundant deepcopy + per-iteration C++ rebuild."""
    result = SimpleEvent.from_data()
    for variable in event.variables:
        for simple_set in event.simple_sets:
            if variable not in result:
                result[variable] = simple_set[variable].__deepcopy__()
            else:
                result[variable] = (
                    result[variable]
                    .__deepcopy__()
                    .union_with(simple_set[variable].__deepcopy__())
                )
    return result


def fill_missing_variables_old(simple_event: SimpleEvent, variables):
    """Original fill_missing_variables: one C++ rebuild per missing variable."""
    for variable in variables:
        if variable not in simple_event:
            simple_event[variable] = variable.domain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(n_simple_sets: int, n_variables: int) -> Event:
    """Build an Event with n_simple_sets simple events, each over n_variables."""
    variables = [Continuous(f"x{i}") for i in range(n_variables)]
    simple_sets = []
    for s in range(n_simple_sets):
        lo, hi = float(s), float(s + 1)
        data = {v: closed(lo, hi) for v in variables}
        simple_sets.append(SimpleEvent.from_data(data))
    return Event.from_simple_sets(*simple_sets)


def make_simple_event_and_missing(n_present: int, n_missing: int):
    """Return a SimpleEvent with n_present variables and a list of n_missing extra variables."""
    present_vars = [Continuous(f"p{i}") for i in range(n_present)]
    missing_vars = [Continuous(f"m{i}") for i in range(n_missing)]
    data = {v: closed(0.0, 1.0) for v in present_vars}
    se = SimpleEvent.from_data(data)
    return se, missing_vars


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

REPEATS = 7        # independent timing runs per configuration
INNER_LOOPS = 50   # calls per timing run (timeit number)


def bench(fn, setup_fn, repeats=REPEATS, number=INNER_LOOPS):
    """
    Time fn() across `repeats` independent runs, each calling fn() `number` times.
    setup_fn() is called before each run to produce a fresh input.
    Returns (mean_ms, stdev_ms) per single call.
    """
    times_ms = []
    for _ in range(repeats):
        obj = setup_fn()
        t = timeit.timeit(lambda: fn(obj), number=number)
        times_ms.append(t / number * 1000)
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def bench_bounding_box():
    print("\n" + "=" * 65)
    print("bounding_box()  — varying #simple_sets × #variables")
    print("=" * 65)
    print(f"{'simple_sets':>12} {'variables':>10} {'old (ms)':>12} {'new (ms)':>12} {'speedup':>9}")
    print("-" * 65)

    configs = [
        (2, 2), (5, 5), (10, 5), (20, 5),
        (5, 10), (5, 20), (10, 10), (20, 10),
    ]

    for n_ss, n_vars in configs:
        event = make_event(n_ss, n_vars)

        old_mean, old_std = bench(
            fn=bounding_box_old,
            setup_fn=lambda: event,
        )
        new_mean, new_std = bench(
            fn=lambda e: e.bounding_box(),
            setup_fn=lambda: event,
        )
        speedup = old_mean / new_mean if new_mean else float("inf")
        print(
            f"{n_ss:>12} {n_vars:>10} "
            f"{old_mean:>10.3f}ms {new_mean:>10.3f}ms "
            f"{speedup:>8.2f}x"
        )


def bench_fill_missing_variables():
    print("\n" + "=" * 65)
    print("fill_missing_variables()  — varying #present × #missing")
    print("=" * 65)
    print(f"{'present':>10} {'missing':>10} {'old (ms)':>12} {'new (ms)':>12} {'speedup':>9}")
    print("-" * 65)

    configs = [
        (1, 1), (1, 5), (1, 10), (1, 20),
        (5, 5), (5, 10), (10, 10), (10, 20),
    ]

    for n_present, n_missing in configs:

        def setup_old():
            se, missing = make_simple_event_and_missing(n_present, n_missing)
            return se, missing

        def setup_new():
            se, missing = make_simple_event_and_missing(n_present, n_missing)
            return se, missing

        def run_old(args):
            se, missing = args
            fill_missing_variables_old(se, missing)

        def run_new(args):
            se, missing = args
            se.fill_missing_variables(missing)

        # Each call mutates the SimpleEvent, so we need a fresh one each time.
        # bench() calls setup_fn() only once per repeat, but the inner loop
        # calls fn() `number` times on the same object. After the first call,
        # fill_missing_variables is a no-op (nothing left to fill). To get a
        # fair measurement of the filling work, we use number=1 and more repeats.
        repeats = 30

        old_times = []
        new_times = []
        for _ in range(repeats):
            se_old, missing = make_simple_event_and_missing(n_present, n_missing)
            t_old = timeit.timeit(lambda: fill_missing_variables_old(se_old, missing), number=1)
            old_times.append(t_old * 1000)

            se_new, missing = make_simple_event_and_missing(n_present, n_missing)
            t_new = timeit.timeit(lambda: se_new.fill_missing_variables(missing), number=1)
            new_times.append(t_new * 1000)

        old_mean = statistics.mean(old_times)
        new_mean = statistics.mean(new_times)
        speedup = old_mean / new_mean if new_mean else float("inf")

        print(
            f"{n_present:>10} {n_missing:>10} "
            f"{old_mean:>10.4f}ms {new_mean:>10.4f}ms "
            f"{speedup:>8.2f}x"
        )


def bench_bounding_box_scaling():
    """Show how speedup scales as simple_sets grows (fixed 5 variables)."""
    print("\n" + "=" * 65)
    print("bounding_box() scaling — #simple_sets (5 variables fixed)")
    print("=" * 65)
    print(f"{'simple_sets':>12} {'old (ms)':>12} {'new (ms)':>12} {'speedup':>9}")
    print("-" * 65)

    for n_ss in [1, 2, 5, 10, 20, 50, 100]:
        event = make_event(n_ss, 5)
        old_mean, _ = bench(fn=bounding_box_old, setup_fn=lambda: event)
        new_mean, _ = bench(fn=lambda e: e.bounding_box(), setup_fn=lambda: event)
        speedup = old_mean / new_mean if new_mean else float("inf")
        print(f"{n_ss:>12} {old_mean:>10.3f}ms {new_mean:>10.3f}ms {speedup:>8.2f}x")


def bench_fill_missing_scaling():
    """Show how speedup scales as #missing grows (1 present variable fixed)."""
    print("\n" + "=" * 65)
    print("fill_missing_variables() scaling — #missing (1 present fixed)")
    print("=" * 65)
    print(f"{'missing':>10} {'old (ms)':>12} {'new (ms)':>12} {'speedup':>9}")
    print("-" * 65)

    for n_missing in [1, 2, 5, 10, 20, 50]:
        old_times, new_times = [], []
        for _ in range(40):
            se_old, missing = make_simple_event_and_missing(1, n_missing)
            t = timeit.timeit(lambda: fill_missing_variables_old(se_old, missing), number=1)
            old_times.append(t * 1000)

            se_new, missing = make_simple_event_and_missing(1, n_missing)
            t = timeit.timeit(lambda: se_new.fill_missing_variables(missing), number=1)
            new_times.append(t * 1000)

        old_mean = statistics.mean(old_times)
        new_mean = statistics.mean(new_times)
        speedup = old_mean / new_mean if new_mean else float("inf")
        print(f"{n_missing:>10} {old_mean:>10.4f}ms {new_mean:>10.4f}ms {speedup:>8.2f}x")


if __name__ == "__main__":
    print("random_events performance benchmark")
    print("Comparing old (pre-optimisation) vs new (optimised) implementations")
    bench_bounding_box()
    bench_fill_missing_variables()
    bench_bounding_box_scaling()
    bench_fill_missing_scaling()
    print()
