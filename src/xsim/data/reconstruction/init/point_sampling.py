import torch
import torch.nn.functional as F
from xsimgs.structures import ROIBox3D


def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def uniform_sample_sphere(num_samples, device, inverse=False):
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = torch.sign(dist) * torch.abs(dist) ** (1. / 3)
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(0.02)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts


def far_points_sampling(
        roi: ROIBox3D,
        num_samples: int,
        far: float = 1e4,
        device: torch.device | str = 'cpu'
):
    random_directions = torch.rand(num_samples, 3, device=device) - 0.5
    random_directions[:, -1] = torch.abs(random_directions[:, -1])
    random_directions = F.normalize(random_directions, dim=-1)

    near = roi.size[:2].amin() * 0.5
    random_distances = torch.rand(num_samples, 1, device=device)
    random_distances = (1 - random_distances) / near + random_distances / far
    return random_directions / random_distances
