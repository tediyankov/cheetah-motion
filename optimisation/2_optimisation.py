import pickle
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
from scipy.optimize import minimize

PROJECT_ROOT = Path("/gws/nopw/j04/iecdt/tyankov/cheetah-motion")
sys.path.insert(0, str(PROJECT_ROOT))

import data.sequence as _seq_module
sys.modules["sequence"] = _seq_module
from data.sequence import FTE_JOINT_NAMES, Sequence, Sample
from evaluation.metrics import (
    report_metrics,
    per_joint_mpjpe,
    per_bone_length_std,
    _project
)
from optimisation.skeleton import BONES, BONE_LENGTHS_M

## config
SEQ_PATH = Path("/gws/nopw/j04/iecdt/cheetah/2019_03_09/lily/flick")
OUT_DIR = SEQ_PATH / "optimisation"
LAMBDA_SMOOTH = 1e2
LAMBDA_BONE = 1e3
MAX_ITER = 500
TOL = 1e-6

## loading data 
def load_seq (seq_path: Path) -> list[Sample]:
    pkl_path = seq_path / "sequence.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"sequence.pkl not found at {pkl_path}. "
            "run sequence.generate_samples() first."
        )
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    print(f"loaded {len(samples)} samples from {pkl_path}")
    return samples

def load_dlt(seq_path: Path) -> np.ndarray:
    dlt_pkl = seq_path / "dlt" / "dlt.pickle"
    if not dlt_pkl.exists():
        raise FileNotFoundError(f"DLT pickle not found at {dlt_pkl}")
    with open(dlt_pkl, "rb") as f:
        dlt = pickle.load(f)
    positions = np.array(dlt, dtype=np.float64) # (T, J, 3)
    print(f"Loaded DLT positions: {positions.shape}")
    return positions

def unpack_sequences (samples: list[Sample]): 
    T = len(samples)
    C = samples[0].detections_2d.shape[0]
    J = samples[0].detections_2d.shape[1]
    detections = np.zeros((T, C, J, 3), dtype=np.float64)
    gt_3d = np.zeros((T, J, 3), dtype=np.float64)

    for t, s in enumerate(samples):
        detections[t] = s.detections_2d
        gt_3d[t] = s.ground_truth_3d

    projections = samples[0].camera_projections.astype(np.float64)
    return detections, projections, gt_3d

## energy

# reproj error
def energy_reproj(X: np.ndarray, detections: np.ndarray, projections: np.ndarray) -> float:
    T, J, _ = X.shape
    C = projections.shape[0]
    total = 0.0

    for c in range(C):
        P = projections[c]
        if np.all(P == 0):
            continue
        for t in range(T):
            w     = detections[t, c, :, 2] 
            valid = w > 0
            if not valid.any():
                continue
            uv_p = _project(X[t], P) 
            uv_d = detections[t, c, :, :2] 
            diff = uv_p[valid] - uv_d[valid] 
            total += (w[valid] * (diff ** 2).sum(axis=1)).sum()

    return total

# temporal smoothness
def energy_smooth(X: np.ndarray) -> float:
    acc = X[2:] - 2 * X[1:-1] + X[:-2] 
    return float((acc ** 2).sum())

# bone length consistency
def energy_bone(X: np.ndarray, bones: list[tuple[int, int]], ref_lengths: np.ndarray) -> float:
    total = 0.0
    for b, (i, k) in enumerate(bones):
        diff = X[:, i, :] - X[:, k, :]
        lengths = np.linalg.norm(diff, axis=1)
        total += ((lengths - ref_lengths[b]) ** 2).sum()
    return total

# combining into total enery functin
def total_energy(x_flat: np.ndarray,
                 shape: tuple,
                 detections: np.ndarray,
                 projections: np.ndarray,
                 bones: list,
                 ref_lengths: np.ndarray,
                 lam_smooth: float,
                 lam_bone: float) -> float:

    X = x_flat.reshape(shape)
    e_r = energy_reproj(X, detections, projections)
    e_s = energy_smooth(X)
    e_b = energy_bone(X, bones, ref_lengths)
    return e_r + lam_smooth * e_s + lam_bone * e_b # TODO: hyperparam tuning og the two lambda hyperparams 

## reference bone length
def estimate_ref_lengths(X_dlt: np.ndarray,
                         bones: list[tuple[int, int]]) -> np.ndarray:
    ref = []
    for i, k in bones:
        lengths = np.linalg.norm(X_dlt[:, i, :] - X_dlt[:, k, :], axis=-1)
        ref.append(np.nanmedian(lengths))
    return np.array(ref)

## visualisation
def plot_improvement(X_dlt: np.ndarray, X_opt: np.ndarray, gt: np.ndarray, bones: list[tuple[int, int]], out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    T = X_dlt.shape[0]

    # per-bone length std
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("bone length std per bone (mm)", fontsize=13)

    for ax, (X, label, colour) in zip(axes, [
        (X_dlt, "DLT (before)","steelblue"),
        (X_opt, "Optimised (after)","tomato"),
    ]):
        stds  = per_bone_length_std(X, bones)
        names = [
            f"{FTE_JOINT_NAMES[i][:6]}–{FTE_JOINT_NAMES[k][:6]}"
            for i, k in bones
        ]
        ax.barh(names, stds, color=colour, alpha=0.8)
        ax.set_xlabel("Std (mm)")
        ax.set_title(label)
        ax.axvline(stds.mean(), color="black", linestyle="--",
                   label=f"mean={stds.mean():.1f}mm")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "bone_length_std.png", dpi=150)
    plt.close()
    print("Saved → bone_length_std.png")

    # traj smoothness 
    fig, ax = plt.subplots(figsize=(12, 4))
    frames  = np.arange(1, T - 1)

    for X, label, colour in [
        (X_dlt, "DLT", "steelblue"),
        (X_opt, "Optimised", "tomato"),
        (gt, "FTE GT", "green"),
    ]:
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        mag = np.linalg.norm(acc, axis=-1).mean(axis=1) * 1000.0 # converting to mm 
        ax.plot(frames, mag, label=label, alpha=0.8)

    ax.set_xlabel("frame")
    ax.set_ylabel("mean acceleration magnitude (mm)")
    ax.set_title("traj smoothness — mean joint acceleration per frame")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_smoothness.png", dpi=150)
    plt.close()
    print("Saved → trajectory_smoothness.png")

    # per-joint MPJPE 
    fig, ax = plt.subplots(figsize=(14, 5))
    x_pos = np.arange(len(FTE_JOINT_NAMES))
    width = 0.35

    for offset, (X, label, colour) in enumerate([
        (X_dlt, "DLT",       "steelblue"),
        (X_opt, "Optimised", "tomato"),
    ]):
        ax.bar(x_pos + offset * width, per_joint_mpjpe(X, gt),
               width, label=label, color=colour, alpha=0.8)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(FTE_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("per joint MPJPE: DLT vs Optimised")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "per_joint_mpjpe.png", dpi=150)
    plt.close()
    print("Saved → per_joint_mpjpe.png")

## MAIN ANALYSIS 

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # loading data
    samples = load_seq(SEQ_PATH)
    detections, projections, gt = unpack_sequences(samples)
    X_dlt = load_dlt(SEQ_PATH)

    T, J = X_dlt.shape[0], X_dlt.shape[1]
    C = projections.shape[0]
    print(f"T={T} frames, J={J} joints, C={C} cameras")

    # clean NaNs before metrics/plots
    nan_frames = np.isnan(X_dlt).any(axis=(1, 2))
    if nan_frames.any():
        print(f"Dropping {nan_frames.sum()} NaN frames from X_dlt before metrics")
    X_dlt_clean = X_dlt[~nan_frames]
    gt_clean = gt[~nan_frames]
    det_clean = detections[~nan_frames]

    # ensure NaN 2D joints are ignored everywhere
    nan_uv = np.isnan(det_clean[..., :2]).any(axis=-1)  
    det_clean[nan_uv, 2] = 0.0                        
    det_clean[nan_uv, 0] = 0.0                         
    det_clean[nan_uv, 1] = 0.0

    SUBSAMPLE = 2  
    X_dlt_clean = X_dlt_clean[::SUBSAMPLE]
    gt_clean = gt_clean[::SUBSAMPLE]
    det_clean = det_clean[::SUBSAMPLE]

    # ref bone lengths
    ref_lengths = estimate_ref_lengths(X_dlt_clean, BONES)
    print(f"\nReference bone lengths from DLT (metres):")
    for b, (i, k) in enumerate(BONES):
        print(f"  {FTE_JOINT_NAMES[i]:<18} → {FTE_JOINT_NAMES[k]:<18}  "
              f"{ref_lengths[b]:.4f} m")

    # metrics (before)
    metrics_dlt = report_metrics(
        "DLT (before)", X_dlt_clean, gt_clean, det_clean, projections, BONES
    )

    # optimise
    print(f"\nStarting L-BFGS-B optimisation  "
          f"(λ_smooth={LAMBDA_SMOOTH}, λ_bone={LAMBDA_BONE}) ...")

    T_clean, J = X_dlt_clean.shape[0], X_dlt_clean.shape[1]
    shape = (T_clean, J, 3)

    pbar = tqdm(total=MAX_ITER, desc="Optimisation", unit="iter")
    def _callback(_xk):
        pbar.update(1)

    MAX_EVAL = 200000
    result = minimize(
        fun=total_energy,
        x0=X_dlt_clean.flatten(),
        args=(shape, det_clean, projections, BONES,
              ref_lengths, LAMBDA_SMOOTH, LAMBDA_BONE),
        method="L-BFGS-B",
        callback=_callback,
        options={"maxiter": MAX_ITER, "maxfun": MAX_EVAL, "ftol": TOL, "disp": False},
    )
    pbar.close()

    print("\nOptimiser status:")
    print("success:", result.success)
    print("message:", result.message)
    print("nit:", result.nit)
    print("nfev:", result.nfev)

    X_opt_clean = result.x.reshape(shape)
    print(f"\noptimisation finished.  success={result.success}"
          f"final energy={result.fun:.6f}")

    # metrics after (use same valid frames)
    metrics_opt = report_metrics(
        "optimised (after)", X_opt_clean, gt_clean, det_clean, projections, BONES
    )

    # saving
    out_pkl = OUT_DIR / "optimised_positions.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "positions": X_opt_clean,
            "nan_mask": nan_frames,
            "metrics_dlt": metrics_dlt,
            "metrics_opt": metrics_opt,
            "lambda_smooth": LAMBDA_SMOOTH,
            "lambda_bone": LAMBDA_BONE,
        }, f)
    print(f"\nsaved optimised positions → {out_pkl}")

    # viz
    plot_improvement(X_dlt_clean, X_opt_clean, gt_clean, BONES, OUT_DIR)
    print("\nDone.")

if __name__ == "__main__":
    main()

