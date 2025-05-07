#!/usr/bin/env bash
set -e

echo "Cleaning up any leftover processes"

# Kill leftover processes quietly
pkill -f rviz2                          >/dev/null 2>&1 || true
pkill -f ros2\ bag\ record              >/dev/null 2>&1 || true
pkill -f ros2\ bag\ play                >/dev/null 2>&1 || true
pkill -f zed_camera.launch.py           >/dev/null 2>&1 || true
pkill -f pipeline_ros_fast.py           >/dev/null 2>&1 || true

for pid in $(pgrep -f ros); do
  if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
    kill "$pid" 2>/dev/null || true
  fi
done

echo "Done Cleaning"

# ── Paths ──────────────────────────────────────────────────────────────
WS=~/ros2_ws
VENV=/home/df/OFFSEG-AGX/.venv
BAG_DIR_BASE=/media/df/data/rosbags

# ── Calculate Next Run Number ───────────────────────────────────────────
echo "Calculating next run number..."
if [ ! -d "$BAG_DIR_BASE" ]; then
  mkdir -p "$BAG_DIR_BASE"
fi

RUN=$(ls -v "$BAG_DIR_BASE" | grep -E '^run[0-9]+' | sed 's/run//' | sort -n | tail -n 1)
if [ -z "$RUN" ]; then
  RUN=1
else
  RUN=$((RUN + 1))
fi

BAG_DIR="$BAG_DIR_BASE/run${RUN}"
mkdir -p "$BAG_DIR"
echo "Run number: $RUN"
echo "Bag directory: $BAG_DIR"

# ── Source env ─────────────────────────────────────────────────────────
source /home/df/.bashrc
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

# ── Launch ZED camera (if not already running) ────────────────────────
if ! ros2 node list | grep -q '/zed/zed_node'; then
  echo "Launching ZED‑X …"
  ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx &  CAM_PID=$!
  sleep 3
else
  echo "ZED node already running."
fi

# ── Launch segmentation node ──────────────────────────────────────────
source "$VENV/bin/activate"
python /home/df/OFFSEG-AGX/Pipeline/pipeline_ros_fast.py --run-number "$RUN" --ros-publish & SEG_PID=$!
sleep 1   # give it a moment to create the publisher

# -- Start visualization --
# rviz2 -d /home/df/.rviz2/offseg.rviz & RVIZ_PID=$!

# ── Record ONLY the raw RGB + segmentation topics ─────────────────────
ros2 bag record \
      /zed/zed_node/rgb/image_rect_color \
      /bisenetv2/segmented_image \
      --output "$BAG_DIR/seg_run${RUN}" \
      --max-cache-size 1024 & BAG_PID=$!

echo "Recording…  (Ctrl‑C to stop)"

# ── Clean shutdown on Ctrl‑C ───────────────────────────────────────────
trap "echo 'Clean Stopping…'; kill $BAG_PID $SEG_PID $CAM_PID $RVIZ_PID; wait" INT
wait $BAG_PID

