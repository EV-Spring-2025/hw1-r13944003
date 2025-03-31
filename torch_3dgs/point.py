import random
from typing import BinaryIO, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch

from  .camera import extract_camera_params


def get_point_clouds(cameras, depths, alphas, rgbs=None):
    """
    Generates a 3D point cloud from camera parameters, depth maps, and optional RGB colors.

    Args:
        cameras: Camera intrinsics and extrinsics.
        depths: Depth maps of shape (N, H, W), where N is the number of images.
        alphas: Binary mask indicating valid depth points.
        rgbs: Optional RGB color values corresponding to depth points.

    Returns:
        PointCloud: A structured point cloud representation with 3D coordinates and color information.
    """
    Hs, Ws, intrinsics, c2ws = extract_camera_params(cameras)
    W, H = int(Ws[0].item()), int(Hs[0].item())
    focalX, focalY = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    assert (depths.shape == alphas.shape)
    coords = []


    # TODO: Compute ray origins and directions for each pixel
    # Hint: You need to use the camera intrinsics (intrinsics) and extrinsics (c2ws)
    # to convert pixel coordinates into world-space rays.
    # rays_o, rays_d = ......
    Rs = c2ws[:, :3, :3]  # (N, 3, 3)
    ts = c2ws[:, :3, 3]   # (N, 3)
    rays_o = ts  # (N, 3)

    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    uv_h = torch.stack([u, v, torch.ones_like(u)], axis=-1).float().cuda()  # (H, W, 3)

    K_inv = torch.linalg.inv(intrinsics[:, :3, :3])  # (N, 3, 3)
    p_c = torch.einsum('nij,hwj->nhwi', K_inv, uv_h)  # (N, H, W, 3)
    rays_d = torch.einsum('nij,nhwj->nhwi', Rs, p_c)  # (N, H, W, 3)
    rays_d /= rays_d.norm(dim=-1, keepdim=True)


    # TODO: Compute 3D world coordinates using depth values
    # Hint: Use the ray equation: P = O + D * depth
    # P: 3D point, O: ray origin, D: ray direction, depth: depth value
    # pts = ......
    pts = rays_o[:, None, None, :] + rays_d * depths[..., None]

    # TODO: Apply the alpha mask to filter valid points
    # Hint: Mask should be applied to both coordinates and RGB values (if provided)
    # mask = ......
    # coords = pts[mask].cpu().numpy()
    mask = (alphas == 1)
    coords = pts[mask].cpu().numpy()

    if rgbs is not None:
        channels = dict(
            R=rgbs[..., 0][mask].cpu().numpy(),
            G=rgbs[..., 1][mask].cpu().numpy(),
            B=rgbs[..., 2][mask].cpu().numpy(),
            A=alphas[mask].cpu().numpy(),
        )
    else:
        channels = {}
    point_cloud = PointCloud(coords, channels)
    
    return point_cloud


def preprocess(data, channel):
    if channel in ["R", "G", "B", "A"]:
        return np.round(data * 255.0)
    return data


class PointCloud:
    def __init__(self, coords: np.ndarray, channels: Dict[str, np.ndarray]) -> None:
        self.coords = coords
        self.channels = channels

    def __repr__(self) -> str:
        str = f"coords:{len(self.coords)} \t channels:{list(self.channels.keys())}"
        return str

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )
        
    def visualize(self, step_size=50):
        coords = self.coords  # (N, 3)
        x, y, z = coords[::step_size, 0], coords[::step_size, 1], coords[::step_size, 2]
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        channels = {k:v for k, v in self.channels.items()}
        colors = np.stack([channels['R'], channels['G'], channels['B'], channels['A']], axis=1)[::step_size, :]
        ax.scatter(x, y, z, c=colors, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Point Cloud Visualization")
        plt.show()
