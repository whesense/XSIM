from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class LidarModel:
    """Fixed hardware parameters of a spinning lidar.

    These describe the sensor itself, not a dataset: a rig assigns a model to
    each of its lidars, and several datasets may share the same model. Nothing
    here can be derived from a recording -- a beam that returns nothing in a
    sweep simply does not appear in it -- so the values are pinned.

    Attributes:
        num_beams: Number of laser beams.
        beam_elevations: Beam elevations in degrees, sorted from the highest
            beam to the lowest, i.e. in range image row order.
        beam_remap: Raw beam index (the ``laser_id`` reported by the sensor) to
            its row in ``beam_elevations``. Raw indices are not ordered by
            elevation, so this permutation is required to build a range image.
        azimuth_resolution: Azimuth step between adjacent range image columns,
            in degrees. This is a binning step, deliberately finer than the
            sensor's native one: beams do not share a column azimuth (a VLP32
            fires in groups, each with its own azimuth offset) and returns are
            not aligned to bin edges, so binning at the native step makes
            neighbouring returns share a cell and overwrite each other. See the
            models below before widening it.
        beam_azimuth_offsets: Azimuth each beam is mounted at relative to the
            encoder, in degrees, in the same row order as ``beam_elevations``.
            Beams of a column therefore do not share an azimuth. Values are as
            published by the vendor, i.e. in the sensor's own clockwise azimuth
            convention -- ``atan2(y, x)`` runs counter-clockwise and sees them
            negated. **Already applied** to the recorded Cartesian points, so
            they are reference data: re-applying them double counts, and is
            worth up to a full 8.4 deg on a VLP32.
        horizontal_beam_divergence: Angular width of a single beam across the
            scan direction, in degrees.
        vertical_beam_divergence: Angular height of a single beam, in degrees.
    """
    num_beams: int
    beam_elevations: torch.Tensor
    beam_remap: torch.Tensor
    beam_azimuth_offsets: torch.Tensor
    azimuth_resolution: float
    horizontal_beam_divergence: float
    vertical_beam_divergence: float

    def __post_init__(self):
        assert self.beam_elevations.shape == (self.num_beams,)
        assert self.beam_remap.shape == (self.num_beams,)
        assert self.beam_azimuth_offsets.shape == (self.num_beams,)
        assert torch.equal(
            self.beam_remap.sort().values, torch.arange(self.num_beams)
        ), 'beam_remap must be a permutation of the beam rows'
        assert (self.beam_elevations.diff() < 0).all(), \
            'beam_elevations must be sorted from the highest beam to the lowest'

    @property
    def num_azimuth_bins(self) -> int:
        return round(360.0 / self.azimuth_resolution)


# Elevations and remapping were measured from the sweeps themselves:
#   beam_elevs = torch.stack([
#       torch.round(elev[raw_sweep.beam_idx == i].rad2deg().median(), decimals=4).float()
#       for i in range(num_beams)
#   ])
#   beam_remap = torch.zeros(num_beams, dtype=torch.long)
#   beam_remap[beam_elevs.sort().indices] = torch.arange(num_beams - 1, -1, -1)
VELODYNE_VLP32 = LidarModel(
    num_beams=32,
    beam_elevations=torch.tensor([
        15.0000,  10.3330,   7.0000,   4.6670,   3.3330,   2.3330,   1.6670,   1.3330,
         1.0000,   0.6670,   0.3330,  -0.0000,  -0.3330,  -0.6670,  -1.0000,  -1.3330,
        -1.6670,  -2.0000,  -2.3330,  -2.6670,  -3.0000,  -3.3330,  -3.6670,  -4.0000,
        -4.6670,  -5.3330,  -6.1480,  -7.2540,  -8.8430, -11.3100, -15.6390, -25.0000
    ]),
    beam_remap=torch.tensor([
        31, 14, 16, 30, 29, 11, 13, 28,
        27, 10, 12, 26, 25,  7,  9, 23,
        24,  6,  8, 22, 21,  4,  5, 19,
        20,  2,  3, 18, 17,  0,  1, 15
    ]),
    # Vendor table, reordered from laser id to row. The lasers form four groups
    # (5/11/11/5 beams at -4.2/-1.4/+1.4/+4.2 deg); recovering these magnitudes
    # from a raw sweep matches the table to 0.03 deg, which is what confirms
    # both the unit (degrees) and that the points already carry them.
    beam_azimuth_offsets=torch.tensor([
        -1.4,  1.4, -1.4,  1.4, -1.4,  1.4, -4.2, -1.4,
         1.4,  4.2, -4.2, -1.4,  1.4,  4.2, -4.2, -1.4,
         1.4,  4.2, -4.2, -1.4,  1.4,  4.2, -4.2, -1.4,
         1.4,  4.2, -1.4,  1.4, -1.4,  1.4, -1.4,  1.4
    ]),
    azimuth_resolution=0.18,
    horizontal_beam_divergence=0.18,
    vertical_beam_divergence=0.09,
)

VELODYNE_VLP16 = LidarModel(
    num_beams=16,
    beam_elevations=torch.tensor([
        15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0,
        -1.0, -3.0, -5.0, -7.0, -9.0, -11.0, -13.0, -15.0
    ]),
    beam_remap=torch.tensor([
        15,  7, 14,  6, 13,  5, 12,  4,
        11,  3, 10,  2,  9,  1,  8,  0
    ]),
    # A VLP16 mounts every laser at the same azimuth: measured per-laser offsets
    # across a raw sweep span 0.025 deg, i.e. nothing beyond noise.
    beam_azimuth_offsets=torch.zeros(16),
    azimuth_resolution=0.18,
    horizontal_beam_divergence=0.18,
    vertical_beam_divergence=0.09,
)
