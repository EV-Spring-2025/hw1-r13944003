import torch


def compute_mean_knn_dist(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Computes the mean k-nearest neighbor (k-NN) distance for each point in a point cloud.

    Args:
        points (torch.Tensor): A tensor of shape (N, D), where N is the number of points 
                               and D is the dimensionality (e.g., 3 for 3D points).
        k (int): The number of nearest neighbors to consider.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the mean distance to the k nearest neighbors for each point.
    """

    diffs = points[:, None, :] - points[None, :, :]
    dists_sq = torch.sum(diffs ** 2, dim=-1)

    # TODO: Ignore self-distances (distance of each point to itself)
    # Hint: Use fill_diagonal_ to set diagonal values to infinity
    # dists_sq.fill_diagonal_(......)
    dists_sq.fill_diagonal_(torch.inf)

    # TODO: Find k-nearest neighbors
    # Hint: Use torch.topk to find the smallest k distances for each point
    # knn_vals, _ = ......
    knn_vals, _ = torch.topk(dists_sq, k, largest=False)

    # TODO: Compute the mean k-NN distance
    # Hint: Average the distances over the k neighbors
    # mean_knn_dist = ......
    mean_knn_dist = knn_vals.mean()
    
    return mean_knn_dist
