import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_3dgs.render import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from torch_3dgs.point import PointCloud
from torch_3dgs.simple_knn import compute_mean_knn_dist
from torch_3dgs.sh_utils import RGB_to_SH


class GaussianModel(nn.Module):
    """
    A trainable 3D Gaussian model.

    Attributes:
        _xyz:        Positions of Gaussians
        _features_dc: DC (degree 0) SH coefficients
        _features_rest: Higher-order SH coefficients
        _rotation:   Quaternion-based rotation
        _scaling:    Log-scale 3D scaling
        _opacity:    Logit opacity
    """

    def __init__(self, sh_degree: int = 3, debug: bool = False):
        super().__init__()
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self._setup_functions()

    def _setup_functions(self):
        def covariance_from_scaling_rotation(scaling, modifier, rotation):
            L = build_scaling_rotation(modifier * scaling, rotation)
            cov = L @ L.transpose(1, 2)
            return strip_symmetric(cov)

        self.scaling_activation = torch.exp
        self.scaling_inverse = torch.log
        self.rotation_activation = F.normalize
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse = inverse_sigmoid
        self.covariance_fn = covariance_from_scaling_rotation

    def create_from_pcd(self, pcd: PointCloud) -> "GaussianModel":
        """
        Initialize Gaussians from a colored point cloud.
        """
        coords = torch.tensor(np.asarray(pcd.coords), dtype=torch.float32, device="cuda")
        colors = torch.tensor(np.asarray(pcd.select_channels(["R", "G", "B"])), dtype=torch.float32, device="cuda") / 255.0
        features = RGB_to_SH(colors)

        print(f"Number of points at initialization: {coords.shape[0]}")

        num_points = coords.shape[0]
        sh_dim = (self.max_sh_degree + 1) ** 2

        sh_features = torch.zeros((num_points, 3, sh_dim), dtype=torch.float32, device="cuda")
        sh_features[:, :3, 0] = features
        sh_features[:, 3:, 1:] = 0.0
        dist2 = torch.clamp_min(compute_mean_knn_dist(coords, k=3), 1e-7)
        scale = torch.log(torch.sqrt(dist2)).unsqueeze(-1).repeat(1, 3)

        rotation = torch.zeros((num_points, 4), dtype=torch.float32, device="cuda")
        rotation[:, 0] = 1.0

        opacity = self.opacity_inverse(0.1 * torch.ones((num_points, 1), dtype=torch.float32, device="cuda"))

        self._xyz = nn.Parameter(coords.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:, :, :1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((num_points,), dtype=torch.float32, device="cuda")

        return self

    @property
    def scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def features(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_fn(self.scaling, scaling_modifier, self._rotation)

    def save_ply(self, path: str):
        from plyfile import PlyData, PlyElement

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rotation], axis=1)
        dtype_full = [(attr, "f4") for attr in self._construct_ply_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def _construct_ply_attributes(self):
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_dc_{i}" for i in range(self._features_dc.shape[1] * self._features_dc.shape[2])]
        names += [f"f_rest_{i}" for i in range(self._features_rest.shape[1] * self._features_rest.shape[2])]
        names.append("opacity")
        names += [f"scale_{i}" for i in range(self._scaling.shape[1])]
        names += [f"rot_{i}" for i in range(self._rotation.shape[1])]
        return names
