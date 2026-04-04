"""Posterior probability computation for Bayesian crowd counting loss."""
import torch
from torch.nn import Module


class Post_Prob(Module):
    def __init__(
        self,
        sigma: float,
        c_size: int,
        stride: int,
        background_ratio: float,
        use_background: bool,
        device: torch.device,
    ):
        super().__init__()
        assert c_size % stride == 0
        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        self.cood = (
            torch.arange(0, c_size, step=stride, dtype=torch.float32, device=device)
            + stride / 2
        )
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(p) for p in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis_chunk, st_size in zip(dis_list, st_sizes):
                if len(dis_chunk) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(
                            torch.min(dis_chunk, dim=0, keepdim=True)[0], min=0.0
                        )
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis_chunk = torch.cat([dis_chunk, bg_dis], 0)
                    dis_chunk = -dis_chunk / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis_chunk)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = [None for _ in points]
        return prob_list
