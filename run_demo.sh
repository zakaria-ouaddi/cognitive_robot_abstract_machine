#!/bin/bash
# run_demo.sh
# Launches a CoraPlex demo with correct PYTHONPATH (strips conflicting ROS install paths).
#
# Usage:
#   bash run_demo.sh coraplex/demos/pr2/pr2_giskard_pick_place_demo.py
#   bash run_demo.sh coraplex/demos/pr2/pr2_giskard_both_arms_demo.py

DEMO_SCRIPT="${1:-coraplex/demos/pr2/pr2_giskard_pick_place_demo.py}"
MONO_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect active virtualenv or fallback to cram-env / system python
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "$HOME/.virtualenvs/cram-env/bin/python" ]; then
    PYTHON="$HOME/.virtualenvs/cram-env/bin/python"
else
    PYTHON="$(command -v python3)"
fi

# Strip /install/ paths from PYTHONPATH so virtualenv packages take priority
CLEAN_PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "/workspace/ros/install" | tr '\n' ':' | sed 's/:$//')

echo "[run_demo] Launching: $DEMO_SCRIPT"
echo "[run_demo] Stripped ROS install paths from PYTHONPATH"

PYTHONPATH="$CLEAN_PYTHONPATH" exec "$PYTHON" "$MONO_REPO/$DEMO_SCRIPT"
