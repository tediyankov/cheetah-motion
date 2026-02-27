import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import random_split

PROJECT_ROOT = Path("/gws/nopw/j04/iecdt/tyankov/cheetah-motion")
sys.path.insert(0, str(PROJECT_ROOT))

import data.sequence as _seq_module
sys.modules["sequence"] = _seq_module

from data.data_loader import make_pkl_list, CheetahSamplesDataset, BASE_ROOT, CSV, SEED
from data.direct_linear_triangulation import triangulate_entire_sequence
from lifting_net.simpleMLP2 import SimpleMLP2
from evaluation.metrics import report_metrics
from optimisation.skeleton import BONES, BONE_LENGTHS_M
from optimisation import reconstruct_optimisation as opt  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- config ----
MODEL_PATH = PROJECT_ROOT / "lifting_net" / "simplemlp2.pt"
BATCH_SIZE = 64
VAL_FRAC = 0.15
TEST_FRAC = 0.2
SUBSAMPLE = 1
LAMBDA_SMOOTH = 0.0
LAMBDA_BONE = 1e3
LAMBDA_DLT = 0.0
MAX_ITER = 300
TOL = 1e-6


def unpack_sequences(samples):
    first = samples[0]

    if hasattr(first, "detections_2d"):
        T = len(samples)
        C = first.detections_2d.shape[0]
        J = first.detections_2d.shape[1]
        detections = np.zeros((T, C, J, 3), dtype=np.float64)
        gt_3d = np.zeros((T, J, 3), dtype=np.float64)
        for t, s in enumerate(samples):
            detections[t] = s.detections_2d
            gt_3d[t] = s.ground_truth_3d
        projections = first.camera_projections.astype(np.float64)
        return detections, projections, gt_3d

    # tuple path: (det, gt, proj) or (det, gt)
    det0, gt0 = first[0], first[1]
    proj0 = first[2] if len(first) > 2 else None

    T = len(samples)
    C = det0.shape[0]
    J = det0.shape[1]
    detections = np.zeros((T, C, J, 3), dtype=np.float64)
    gt_3d = np.zeros((T, J, 3), dtype=np.float64)

    for t, s in enumerate(samples):
        det, gt = s[0], s[1]
        detections[t] = np.asarray(det)
        gt_3d[t] = np.asarray(gt)

    if proj0 is None:
        raise ValueError("No projections in dataset; cannot compute reprojection metrics.")
    projections = np.asarray(proj0)

    return detections, projections, gt_3d


def clean_detections(detections):
    det = detections.copy()

    nan_uv = np.isnan(det[..., :2]).any(axis=-1)
    det[nan_uv, 2] = 0.0
    det[nan_uv, 0] = 0.0
    det[nan_uv, 1] = 0.0

    valid_views = det[..., 2] > 0
    valid_count = valid_views.sum(axis=1)  # (T, J)
    bad_joint = valid_count < 2
    bad_joint_mask = bad_joint[:, None, :, None]
    det = np.where(bad_joint_mask, 0.0, det)

    return det


def run_mlp_on_sequence(model, detections):
    # detections: (T, C, J, 3)
    preds = []
    with torch.no_grad():
        for i in range(0, detections.shape[0], BATCH_SIZE):
            x = torch.from_numpy(detections[i:i + BATCH_SIZE]).float().to(DEVICE)
            p = model(x).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds, axis=0)


def optimise_sequence(detections, projections, gt):
    # Clean and subsample
    det_clean = clean_detections(detections)

    X_dlt = triangulate_entire_sequence([
        type("S", (), {
            "detections_2d": detections[t],
            "camera_projections": projections,
            "ground_truth_3d": gt[t], 
            "frame_idx": t
        })() for t in range(detections.shape[0])
    ])

    valid = np.isfinite(X_dlt).all(axis=-1)
    X_dlt_clean = X_dlt.copy()
    gt_clean = gt.copy()
    X_dlt_clean[~valid] = np.nan
    gt_clean[~valid] = np.nan

    X_init = opt._fill_nan_linear(X_dlt_clean)

    if SUBSAMPLE > 1:
        X_dlt_clean = X_dlt_clean[::SUBSAMPLE]
        gt_clean = gt_clean[::SUBSAMPLE]
        det_clean = det_clean[::SUBSAMPLE]
        X_init = X_init[::SUBSAMPLE]
        valid = valid[::SUBSAMPLE]

    ref_lengths = np.array(BONE_LENGTHS_M, dtype=np.float64)
    bone_mask = np.isfinite(ref_lengths)

    T_clean, J = X_dlt_clean.shape[0], X_dlt_clean.shape[1]
    shape = (T_clean, J, 3)

    result = opt.minimize(
        fun=opt.total_energy,
        x0=X_init.flatten(),
        args=(shape, det_clean, projections, BONES,
              ref_lengths, LAMBDA_SMOOTH, LAMBDA_BONE, bone_mask,
              X_dlt_clean, valid, LAMBDA_DLT),
        method="L-BFGS-B",
        options={"maxiter": MAX_ITER, "ftol": TOL, "disp": False},
    )

    X_opt = result.x.reshape(shape)
    return X_dlt_clean, X_opt, gt_clean, det_clean


def avg_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def main():
    # build same split as training
    pkl_list = make_pkl_list(CSV, BASE_ROOT)
    dataset = CheetahSamplesDataset(pkl_list, preload=True)

    n = len(dataset)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    n_train = n - n_val - n_test

    torch.manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    # load MLP
    model = SimpleMLP2(num_joints=20, num_cams=6, hidden_size=1024).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    mlp_metrics = []
    dlt_metrics = []
    opt_metrics = []

    for seq in test_ds:
        # seq is a list[Sample]
        detections, projections, gt = unpack_sequences(seq)

        # --- MLP ---
        pred_mlp = run_mlp_on_sequence(model, detections)
        m_mlp = report_metrics("MLP", pred_mlp, gt, detections, projections, BONES)
        mlp_metrics.append(m_mlp)

        # --- DLT + Optimisation ---
        X_dlt, X_opt, gt_clean, det_clean = optimise_sequence(detections, projections, gt)
        m_dlt = report_metrics("DLT", X_dlt, gt_clean, det_clean, projections, BONES)
        m_opt = report_metrics("Optimised", X_opt, gt_clean, det_clean, projections, BONES)
        dlt_metrics.append(m_dlt)
        opt_metrics.append(m_opt)

    print("\n=== AVERAGE TEST METRICS (across test sequences) ===")
    print("MLP:", avg_metrics(mlp_metrics))
    print("DLT:", avg_metrics(dlt_metrics))
    print("Optimised:", avg_metrics(opt_metrics))


if __name__ == "__main__":
    main()