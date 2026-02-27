
# training script for MLP2

import torch
from torch import optim
import sys
from pathlib import Path

PROJECT_ROOT = Path("/gws/nopw/j04/iecdt/tyankov/cheetah-motion")
sys.path.insert(0, str(PROJECT_ROOT))

import data.sequence as _seq_module
sys.modules["sequence"] = _seq_module

from data.data_loader import get_data_loaders
from lifting_net.simpleMLP2 import SimpleMLP2, mpjpe_torch

import numpy as np
from evaluation.metrics import report_metrics
from data.sequence import FTE_JOINT_NAMES

from optimisation.skeleton import BONES, BONE_LENGTHS_M

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def batch_to_tensor(batch):
#     # default collate returns tuple of tensors: (x_batch, y_batch)
#     if isinstance(batch, (list, tuple)) and len(batch) == 2 and torch.is_tensor(batch[0]):
#         return batch[0].float(), batch[1].float()

#     # fallback if a list of samples slipped through
#     xs = torch.stack([torch.from_numpy(s.detections_2d) for s in batch], dim=0).float()
#     ys = torch.stack([torch.from_numpy(s.ground_truth_3d) for s in batch], dim=0).float()
#     return xs, ys

def batch_to_tensor(batch):
    #def batch_to_tensor(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3 and torch.is_tensor(batch[0]):
        x, y, proj = batch
        return x.float(), y.float(), (proj.float() if proj is not None else None)

    xs = torch.stack([torch.from_numpy(s.detections_2d) for s in batch], dim=0).float()
    ys = torch.stack([torch.from_numpy(s.ground_truth_3d) for s in batch], dim=0).float()
    projs = torch.stack([torch.from_numpy(s.camera_projections) for s in batch], dim=0).float()
    return xs, ys, projs

def bone_length_consistency_loss(pred, bones, target_lengths_m, weight=1.0):
    if not bones:
        return torch.tensor(0.0, device=pred.device)

    lengths = []
    for (i, k) in bones:
        lengths.append(torch.norm(pred[:, i, :] - pred[:, k, :], dim=-1))  # (T,)
    lengths = torch.stack(lengths, dim=1)  # (T, nbones)

    target = torch.from_numpy(target_lengths_m).to(pred.device).float().unsqueeze(0)
    return weight * torch.abs(lengths - target).mean()


def combined_loss(pred, gt, w_mpjpe=1.0, w_bone=0.01):
    l_mpjpe = mpjpe_torch(pred, gt)
    l_bone = bone_length_consistency_loss(pred, BONES, BONE_LENGTHS_M, weight=1.0)
    return w_mpjpe * l_mpjpe + w_bone * l_bone

def run_epoch(loader, model, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x, y, _ = batch_to_tensor(batch)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if not torch.isfinite(y).all():
            continue

        if is_train:
            optimizer.zero_grad()

        pred = model(x)
        loss = combined_loss(pred, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.shape[0]
        n += x.shape[0]

    return total_loss / max(n, 1)


def eval_and_report(loader, model, label="val", max_batches=10):
    model.eval()
    preds, gts, dets = [], [], []
    proj_ref = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x, y, proj = batch_to_tensor(batch)
            pred = model(x.to(DEVICE)).cpu().numpy()
            preds.append(pred)
            gts.append(y.numpy())
            dets.append(x.numpy())
            if proj_ref is None:
                proj_ref = proj[0].numpy()  # (C,3,4)

    if not preds:
        return

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    det_np = np.concatenate(dets, axis=0)
    proj_np = proj_ref

    report_metrics(label, pred_np, gt_np, det_np, proj_np, BONES)

def main():
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32, num_workers=4)

    model = SimpleMLP2(num_joints=20, num_cams=6, hidden_size=1024).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30
    for ep in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, model, optimizer)
        val_loss = run_epoch(val_loader, model, optimizer=None)
        print(f"Epoch {ep:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        eval_and_report(val_loader, model, label=f"val@{ep}", max_batches=10)

    test_loss = run_epoch(test_loader, model, optimizer=None)
    print(f"test loss: {test_loss:.4f}")
    eval_and_report(test_loader, model, label="test", max_batches=20)

    out_path = PROJECT_ROOT / "lifting_net" / "simplemlp2.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))

if __name__ == "__main__":
    main()