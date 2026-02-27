
# code for simple MLP predicting 3D joint position from 2D coords for a single frame
# input: 2D coords tensor (C, J, 2)
# output: 3D coords tensor (J, 3)
# needs to optimise a mpjpe loss between predicted 3D and ground truth (from the sequence)
# from Martinez paper -> we implement it using batchnorm and residual connections

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        return out + res


class SimpleMLP2(torch.nn.Module):

    def __init__(self, num_joints=20, num_cams=6, hidden_size=1024, num_blocks=3, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.num_cams = num_cams
        self.input_size = num_cams * num_joints * 2
        self.output_size = num_joints * 3

        self.fc_in = nn.Linear(self.input_size, hidden_size)
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x[..., :2]  # (T, C, J, 2)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        T, C, J, _ = x.shape
        assert C == self.num_cams and J == self.num_joints, \
            f"expected (C,J)=({self.num_cams},{self.num_joints}) got ({C},{J})"

        x = x.reshape(T, -1)

        out = self.fc_in(x)
        out = self.bn_in(out)
        out = torch.relu(out)

        for blk in self.blocks:
            out = blk(out)

        out = self.fc_out(out)
        out = out.view(T, J, 3)
        return out if T > 1 else out.squeeze(0)

def mpjpe_torch(pred, gt):
    """
    pred, gt: (J,3) or (T,J,3)
    returns mean per-joint position error
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    return torch.norm(pred - gt, dim=-1).mean()































