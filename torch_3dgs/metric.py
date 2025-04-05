import torch
import torch.nn.functional as F


EPSILON = 1e-6


def calc_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        return torch.sum((x - y) ** 2 * mask.unsqueeze(-1)) / (
            mask.sum() * x.shape[-1] + EPSILON
        )
    return ((x - y) ** 2).mean()

def calc_l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sum_square = ((x-y)**2).sum()
    L = x.shape[0]
    return sum_square**(1/2) / L

def calc_psnr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    return -10 * torch.log10(calc_mse(x, y, mask) + EPSILON)


def calc_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:

    def gaussian_1d_kernel(window_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(window_size: int, channel: int, dtype, device) -> torch.Tensor:
        _1d = gaussian_1d_kernel(window_size, 1.5).unsqueeze(1)
        _2d = _1d @ _1d.T
        window = _2d.float().unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(dtype=dtype, device=device)

    channel = img1.size(-3)
    window = create_window(window_size, channel, img1.dtype, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))
