# Waymo Open Dataset setup

XSIM reads the [Waymo Open Dataset](https://waymo.com/open/) in its **v2
(parquet) format**. Download a split (e.g. `training`) following Waymo's
instructions, then point the path variables at it in
`configs/data/dataset_paths.yaml`

## Downloading single scenes

A full split is over a terabyte, and in the v2 layout one scene is spread
across the per-component folders of the bucket — there is no per-scene folder
to grab. To try XSIM on a few scenes without downloading everything, use the
helper script, which fetches just the components XSIM reads for one scene
(a few GB) into the expected layout:

```bash
# by scene_idx (resolved through configs/data/waymo/scene_names.yaml)...
tools/data/download_waymo_scene.sh 788 /path/to/waymo/training
# ...or by full segment name
tools/data/download_waymo_scene.sh 9653249092275997647_980_000_1000_000 /path/to/waymo/training
```

With `WAYMO_ROOT_PATH` exported the destination argument can be omitted. The
script needs the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
authenticated (`gcloud auth login`) with a Google account that has accepted
the Waymo Open Dataset license.

## Path variables

- `WAYMO_ROOT_PATH` - the v2 split directory (e.g. `.../v2/training`).
- `WAYMO_CACHE_PATH` - where XSIM writes its preprocessed cache. Built
  automatically the first time a scene is used; subsequent runs read the cache.
- `WAYMO_SMPL_DATA_PATH` - a format string resolving to per-scene
  human pose pickles (e.g. `"<poses_root>/{:03d}/humanpose/smpl.pkl"`,
  drivestudio format). Needed for reconstructing pedestrians. 
  Preprocessed poses for the Waymo scenes can be downloaded
  [here](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j)
  (the download link provided by the
  [drivestudio](https://github.com/ziyc/drivestudio) repository).

Scenes are selected by index within the split via `provider.scene_idx` (see
`configs/data/waymo/provider.yaml`, overridable from the command line):
