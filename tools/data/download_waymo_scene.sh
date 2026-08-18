#!/usr/bin/env bash
# Download a single Waymo Open Dataset v2 scene instead of the whole split.
#
# Fetches only the components XSIM reads (the segment's parquet file from each
# component folder) into the layout the provider expects:
#   <dest>/<component>/<segment>.parquet
#
# Requires the gcloud CLI, authenticated with a Google account that has
# accepted the Waymo Open Dataset license (gcloud auth login).
#
# Usage:
#   tools/data/download_waymo_scene.sh <scene> [dest] [split]
#
#   scene  scene_idx from configs/data/waymo/scene_names.yaml (e.g. 788),
#          or a full segment name (e.g. 10017090168044687777_6380_000_6400_000)
#   dest   target directory; defaults to $WAYMO_ROOT_PATH
#   split  bucket split, default: training
set -euo pipefail

BUCKET="gs://waymo_open_dataset_v_2_0_1"
COMPONENTS=(
    camera_image
    camera_calibration
    lidar
    lidar_calibration
    lidar_pose
    lidar_box
    lidar_segmentation
    vehicle_pose
)

scene="${1:?usage: $0 <scene_idx|segment_name> [dest] [split]}"
dest="${2:-${WAYMO_ROOT_PATH:?pass a dest dir or export WAYMO_ROOT_PATH}}"
split="${3:-training}"

# a short number is a scene_idx: resolve it through the scene names list,
# which is a flat ordered yaml list ("  - '<segment>'" lines) under train:
if [[ "$scene" =~ ^[0-9]{1,3}$ ]]; then
    scene_names="$(dirname "$0")/../../configs/data/waymo/scene_names.yaml"
    segment=$(grep -o "'[^']*'" "$scene_names" | sed -n "$((scene + 1))p" | tr -d "'")
    if [[ -z "$segment" ]]; then
        echo "scene_idx $scene not found in $scene_names" >&2
        exit 1
    fi
    echo "scene_idx $scene -> $segment"
else
    segment="$scene"
fi

for component in "${COMPONENTS[@]}"; do
    mkdir -p "$dest/$component"
    echo "downloading $component/$segment.parquet"
    # tolerate a missing bucket file so one unavailable component (or a
    # transient failure) doesn't abort the rest of the scene
    if ! gcloud storage cp "$BUCKET/$split/$component/$segment.parquet" "$dest/$component/"; then
        echo "warning: failed to fetch $component for this segment, skipped" >&2
    fi
done

echo "Scene stored in $dest"
