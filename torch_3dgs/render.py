import math
import torch
import torch.nn.functional as F
from torch import nn

from .sh_utils import eval_sh


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def homogeneous(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float32, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


def build_covariance_3d(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    L = build_scaling_rotation(s, r)
    return L @ L.transpose(1, 2)


def strip_symmetric(cov: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((cov.shape[0], 6), dtype=torch.float32, device=cov.device)
    out[:, 0] = cov[:, 0, 0]
    out[:, 1] = cov[:, 0, 1]
    out[:, 2] = cov[:, 0, 2]
    out[:, 3] = cov[:, 1, 1]
    out[:, 4] = cov[:, 1, 2]
    out[:, 5] = cov[:, 2, 2]
    return out


def projection_ndc(
    points: torch.Tensor,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    points_o = homogeneous(points)
    points_h = points_o @ view_matrix @ proj_matrix
    w_inv = 1.0 / (points_h[..., -1:] + 1e-6)
    p_proj = points_h * w_inv
    p_view = points_o @ view_matrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


def build_covariance_2d(
    mean3d: torch.Tensor,
    cov3d: torch.Tensor,
    view_matrix: torch.Tensor,
    fov_x: float,
    fov_y: float,
    focal_x: float,
    focal_y:float,
) -> torch.Tensor:

    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    cam_space = mean3d @ view_matrix[:3, :3] + view_matrix[-1:, :3]

    tx = (cam_space[..., 0] / cam_space[..., 2]).clamp(-tan_fovx * 1.3, tan_fovx * 1.3) * cam_space[..., 2]
    ty = (cam_space[..., 1] / cam_space[..., 2]).clamp(-tan_fovy * 1.3, tan_fovy * 1.3) * cam_space[..., 2]
    tz = cam_space[..., 2]

    J = torch.zeros(mean3d.shape[0], 3, 3, device=mean3d.device)
    J[..., 0, 0] = focal_x / tz
    J[..., 0, 2] = -tx * focal_x / (tz * tz)
    J[..., 1, 1] = focal_y / tz
    J[..., 1, 2] = -ty * focal_y / (tz * tz)

    W = view_matrix[:3, :3].T
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)
    return cov2d[:, :2, :2] + torch.eye(2, device=cov2d.device)[None] * 0.3


def get_radius(cov2d: torch.Tensor) -> torch.Tensor:
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2 - det).clamp(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2 - det).clamp(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_rect(
    pix_coord: torch.Tensor,
    radii: torch.Tensor,
    width: float,
    height: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    rect_min = (pix_coord - radii[:, None])
    rect_max = (pix_coord + radii[:, None])
    rect_min[..., 0] = rect_min[..., 0].clamp(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clamp(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clamp(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clamp(0, height - 1.0)
    return rect_min, rect_max


class GaussRenderer(nn.Module):
    def __init__(
        self,
        active_sh_degree: int = 3,
        tile_size: int = 50,
    ) -> None:

        super().__init__()
        self.active_sh_degree = active_sh_degree
        self.tile_size = tile_size
        self.pix_coord = None
        self.debug = False

    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0, 2, 1), rays_d)
        return (color + 0.5).clamp(min=0.0)

    def render(self, camera, means2D, cov2d, color, opacity, depths):
        radii = get_radius(cov2d)
        
        rect_min, rect_max = get_rect(means2D, radii, camera.image_width, camera.image_height)

        if self.pix_coord is None:
            self.pix_coord = torch.stack(
                torch.meshgrid(
                    torch.arange(camera.image_width),
                    torch.arange(camera.image_height),
                    indexing="xy"
                ), dim=-1).to("cuda")

        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to("cuda")
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to("cuda")
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to("cuda")

        for h in range(0, camera.image_height, self.tile_size):
            for w in range(0, camera.image_width, self.tile_size):
                over_tl = rect_min[..., 0].clamp(min=w), rect_min[..., 1].clamp(min=h)
                over_br = rect_max[..., 0].clamp(max=w+self.tile_size-1), rect_max[..., 1].clamp(max=h+self.tile_size-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.any():
                    continue
                
                # TODO: Extract the pixel coordinates for this tile.
                # Hint: The tile's pixel coordinates should be extracted using slicing and flattening.
                # tile_coord = ...
                tile_coord = self.pix_coord[h:h+self.tile_size, w:w+self.tile_size].reshape(-1, 2)
                
    
                # TODO: Sort Gaussians by depth.
                # Hint: Sorting should be based on the depth values of Gaussians.
                # sorted_depths, index = ...
                sorted_depths, index = torch.sort(depths[in_mask])
    
                # TODO: Extract relevant Gaussian properties for the tile.
                # Hint: Use the computed index to rearrange the following tensors.
                # sorted_means2D = ...
                # sorted_cov2d = ...
                # sorted_conic = ...
                # sorted_opacity = ...
                # sorted_color = ...
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index]
                sorted_conic = torch.inverse(sorted_cov2d)
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
    
                # TODO: Compute the distance from each pixel in the tile to the Gaussian centers.
                # Hint: This involves computing dx, dy between pixel coordinates and Gaussian centers.
                # dx = ...
                # dx_0, dx_1 = ...
                dx = tile_coord[:, None, :] - sorted_means2D[None, :, :]
    
                # TODO: Compute the 2D Gaussian weight for each pixel.
                # Hint: The weight is determined by the Mahalanobis distance using the covariance matrix.
                # gauss_weight = ...
                step1 = torch.einsum('pni,nij->pnj', dx, sorted_conic)
                quad_term = torch.einsum('pnj,pnj->pn', step1, dx)
                gauss_weight = torch.exp(-0.5 * quad_term)
                # TODO: Compute the alpha blending using transmittance (T).
                # Hint: Ensure proper transparency blending by applying the alpha compositing formula.
                # alpha = ...
                # T = ...
                # acc_alpha = ...
                alpha = 1 - torch.exp(-sorted_opacity)
                weighted_alpha = torch.einsum('pn,n->pn', gauss_weight, alpha.squeeze(-1))
                T = torch.cumprod(1.0 - weighted_alpha + 1e-6, dim=1)  # (num_pixel, num_blob)
                acc_alpha = (1.0 - T[:, -1]).reshape(self.tile_size, self.tile_size)
    
                # TODO: Compute the color and depth contributions.
                # Hint: Perform weighted summation using computed transmittance and opacity.
                # tile_color = ...
                # tile_depth = ...
                T_prev = torch.cat([torch.ones((T.shape[0], 1), device=T.device), T[:, :-1]], dim=1)
                white_background = torch.Tensor([1, 1, 1]).cuda()
                tile_color = torch.einsum('pn,pn,nc->pc', T_prev, weighted_alpha, sorted_color) + torch.einsum('p,c->pc', 1- torch.sum(T_prev*weighted_alpha, dim=-1), white_background)
                tile_color = tile_color.reshape(self.tile_size, self.tile_size, -1)
                tile_depth = torch.einsum('pn,pn,n->p', T_prev, weighted_alpha, sorted_depths).reshape(self.tile_size, self.tile_size)
    
                # TODO: Store computed values into rendering buffers.
                # Hint: Assign tile-wise computed values to corresponding locations in the full image buffers.
                # self.render_color[h:h+self.tile_size, w:w+self.tile_size] = ...
                # self.render_depth[h:h+self.tile_size, w:w+self.tile_size] = ...
                # self.render_alpha[h:h+self.tile_size, w:w+self.tile_size] = ...
                self.render_color[h:h+self.tile_size, w:w+self.tile_size] = tile_color
                self.render_depth[h:h+self.tile_size, w:w+self.tile_size] = tile_depth.unsqueeze(-1)
                self.render_alpha[h:h+self.tile_size, w:w+self.tile_size] = acc_alpha.unsqueeze(-1)

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def forward(self, camera, pc):
        mean_ndc, mean_view, in_mask = projection_ndc(pc.xyz, camera.world_to_view, camera.projection)
        assert in_mask.any(), "No points in frustum"
        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        depths = mean_view[:, 2]

        color = self.build_color(pc.xyz, pc.features, camera)
        cov3d = build_covariance_3d(pc.scaling, pc.rotation)
        cov2d = build_covariance_2d(
            mean3d=pc.xyz,
            cov3d=cov3d,
            view_matrix=camera.world_to_view,
            fov_x=camera.focal_x,
            fov_y=camera.focal_y,
            focal_x=camera.focal_x,
            focal_y=camera.focal_y
        )
        
        color = color[in_mask]
        cov2d = cov2d[in_mask]

        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        return self.render(camera, means2D, cov2d, color, pc.opacity, depths)
