from dataclasses import dataclass

import torch
import torch.nn.functional as F

from xsimgs.structures import ROIBox3D

from xsim.structures import Sweep


@dataclass
class GroundPlaneParams:
    num_iterations: int = 1000
    # Max plane-point distance to consider as inlier
    inlier_threshold: float = 0.1
    # Minimum spread of the three sampled points, so that a near-degenerate
    # triple does not define a wildly tilted plane
    min_sample_distance: float = 2.0
    random_seed: int = 42

    # Points more than this far below the plane are dropped. Negative: the
    # tolerance is downward, and generous enough to keep real dips in the road.
    # Applied after the fit, so it is not part of `fit_key`.
    min_filter_height: float = -0.8
    # Cap on the points RANSAC scores against. A full scene runs to tens of
    # millions of returns, and the fit is unchanged by scoring a large random
    # subset of them.
    max_fit_points: int = 2_000_000

    def fit_key(self) -> dict:
        """The parameters the fitted plane actually depends on.

        Keyed on for caching, so that changing only the filtering tolerance
        reuses the cached plane instead of refitting for nothing.
        """
        return {
            name: getattr(self, name) for name in (
                'num_iterations', 'inlier_threshold', 'min_sample_distance',
                'random_seed', 'max_fit_points'
            )
        }


def fit_plane(
        points: torch.Tensor,
        max_iterations: int = 500,
        inlier_threshold: float = 0.3,
        min_sample_distance: float = 2.0,
        random_seed: int = 42
) -> tuple[torch.Tensor, int]:
    """RANSAC plane fit.

    Args:
        points: ``[N, 3]`` point cloud to fit the plane to.
        max_iterations: Number of RANSAC iterations.
        inlier_threshold: Largest plane-point distance still counted as inlier.
        min_sample_distance: Minimum distance between the three sampled points
            for the sample to be considered.
        random_seed: Seed for the sampling.

    Returns:
        The plane as ``(nx, ny, nz, w)`` with the normal pointing up, and the
        number of inliers supporting it.
    """
    num_points = len(points)
    best_inliers = torch.zeros(num_points, device=points.device, dtype=torch.bool)
    min_error = torch.inf

    generator = torch.Generator().manual_seed(random_seed)

    for _ in range(max_iterations):
        first, second, third = points[
            torch.randint(num_points, size=(3,), generator=generator)
        ]
        min_dist = min(
            (first - second).norm(),
            min((second - third).norm(), (first - third).norm())
        ).item()
        if min_dist < min_sample_distance:
            continue

        normal = F.normalize(
            torch.cross(second - first, third - first, dim=0), dim=0
        )
        if normal[2] < 0:
            normal = -normal

        offset = -torch.dot(normal, first)
        distances = ((points * normal.view(1, 3)).sum(dim=1) + offset).abs()

        inliers = distances < inlier_threshold
        error = (~inliers).sum()
        if error < min_error:
            best_inliers = inliers
            min_error = error

    # Refit on the consensus set: the sampled triple only nominates a plane.
    support = points[best_inliers]
    solution = torch.linalg.lstsq(
        support, torch.ones(len(support), device=points.device), rcond=-1
    )[0]
    if solution[2] < 0:
        solution = -solution

    plane = torch.cat([
        F.normalize(solution, dim=0), (1.0 / solution.norm()).view(1)
    ], dim=0)

    return plane, int(best_inliers.sum())


def plane_point_mask(
        points: torch.Tensor,
        plane: torch.Tensor,
        min_height: float
) -> torch.Tensor:
    """Which points sit above a plane, within a tolerance.

    Args:
        points: ``[N, 3]`` points to test.
        plane: ``(nx, ny, nz, w)`` plane equation.
        min_height: Signed offset from the plane. Negative values keep points
            below it -- ``-0.5`` still accepts a point 50 cm under the ground.

    Returns:
        Boolean mask of the points to keep.
    """
    homogeneous = torch.cat([
        points, torch.ones(len(points), 1, device=points.device)
    ], dim=1)
    signed_distance = (homogeneous * plane.view(1, 4).to(points.device)).sum(dim=1)

    return signed_distance > min_height


def fit_ground_plane(
        sweeps: list[Sweep],
        cfg: GroundPlaneParams,
        device: str | torch.device = 'cuda'
) -> torch.Tensor:
    """Fit one ground plane to every return of a scene.

    Args:
        sweeps: Every sweep of the scene, in any order; only masked returns are
            used.
        cfg: Fitting parameters.
        device: Where to run the fit.

    Returns:
        The plane as ``(nx, ny, nz, w)``, on the CPU.
    """
    points = torch.cat([sweep.masked.xyz.view(-1, 3) for sweep in sweeps], dim=0)
    points = points.to(device)
    if len(points) > cfg.max_fit_points:
        # Seeded, or the fit would differ run to run despite RANSAC's own seed.
        generator = torch.Generator().manual_seed(cfg.random_seed)
        keep = torch.randperm(len(points), generator=generator)[:cfg.max_fit_points]
        points = points[keep.to(points.device)]

    plane, _ = fit_plane(
        points,
        max_iterations=cfg.num_iterations,
        inlier_threshold=cfg.inlier_threshold,
        min_sample_distance=cfg.min_sample_distance,
        random_seed=cfg.random_seed
    )

    return plane.cpu()


def filter_sweeps_by_ground_plane(
        sweeps: list[Sweep],
        plane: torch.Tensor,
        min_height: float
) -> int:
    """Clear the mask of returns that lie below the ground, in place.

    Returns:
        How many returns were dropped.
    """
    removed = 0
    for sweep in sweeps:
        above = plane_point_mask(
            sweep.xyz.view(-1, 3), plane, min_height=min_height
        ).view(*sweep.shape)
        removed += int((~above & sweep.mask).sum())
        sweep.mask &= above

    return removed


def ground_plane_mesh(roi: ROIBox3D, plane: torch.Tensor):
    """The ground plane as a quad spanning an ROI, for visualization."""
    from plytorch import Mesh

    plane = plane.cpu()
    corners = torch.tensor([
        [roi.vmin[0], roi.vmin[1]],
        [roi.vmin[0], roi.vmax[1]],
        [roi.vmax[0], roi.vmin[1]],
        [roi.vmax[0], roi.vmax[1]],
    ], dtype=plane.dtype)
    # Solve the plane equation for z at each corner.
    height = -(
        corners[:, 0] * plane[0] + corners[:, 1] * plane[1] + plane[3]
    ) / plane[2]

    return Mesh(
        points=torch.cat([corners, height.view(-1, 1)], dim=1),
        faces=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int)
    )
