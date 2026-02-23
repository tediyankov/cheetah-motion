import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sequence import FTE_JOINT_NAMES


# projection helper
def _project(X_j: np.ndarray, P: np.ndarray) -> np.ndarray:
    J = X_j.shape[0]
    Xh = np.hstack([X_j, np.ones((J, 1))])  
    p = (P @ Xh.T).T 
    return p[:, :2] / p[:, 2:3]  

## MJPE (mm)

# overall
def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
    return float(np.linalg.norm(pred - gt, axis=-1).mean()) * 1000.0

# per joint
def per_joint_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=0) * 1000.0 

## PA-MPJPE (mm)

# procrustes alignment
def _procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        return pred

    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    p = pred - mu_p
    g = gt   - mu_g

    norm_p = np.linalg.norm(p)
    norm_g = np.linalg.norm(g)
    if norm_p < 1e-8 or norm_g < 1e-8:
        return pred

    scale = norm_g / norm_p
    p = p * scale

    try:
        U, _, Vt = np.linalg.svd(g.T @ p)
    except np.linalg.LinAlgError:
        return pred

    return p @ (U @ Vt).T + mu_g


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    assert pred.shape == gt.shape
    errors = []
    n_skipped = 0

    for t in range(pred.shape[0]):
        if not (np.isfinite(pred[t]).all() and np.isfinite(gt[t]).all()):
            n_skipped += 1
            continue
        aligned = _procrustes_align(pred[t], gt[t])
        errors.append(np.linalg.norm(aligned - gt[t], axis=-1).mean())

    if n_skipped > 0:
        print(f"[pa_mpjpe] skipped {n_skipped}/{pred.shape[0]} NaN frames")

    return float(np.mean(errors)) * 1000.0 if errors else float("nan")

## Bone length standard deviation

# overall
def bone_length_std(X: np.ndarray,
                    bones: list[tuple[int, int]]) -> float:
    stds = []
    for i, k in bones:
        lengths = np.linalg.norm(X[:, i, :] - X[:, k, :], axis=-1) * 1000.0
        stds.append(lengths.std())
    return float(np.mean(stds))

# per bone
def per_bone_length_std(X: np.ndarray,
                        bones: list[tuple[int, int]]) -> np.ndarray:
    stds = []
    for i, k in bones:
        lengths = np.linalg.norm(X[:, i, :] - X[:, k, :], axis=-1) * 1000.0
        stds.append(lengths.std())
    return np.array(stds)

## Mean projection error
def mean_reproj_error(X: np.ndarray,
                      detections: np.ndarray,
                      projections: np.ndarray) -> float:
    T, J, _ = X.shape
    C = projections.shape[0]
    errors = []

    for c in range(C):
        P = projections[c]
        if np.all(P == 0):
            continue
        for t in range(T):
            w = detections[t, c, :, 2]
            valid = (w > 0) & np.isfinite(detections[t, c, :, 0]) & np.isfinite(detections[t, c, :, 1])
            if not valid.any():
                continue
            uv_p = _project(X[t], P)
            uv_d = detections[t, c, :, :2]
            diff = np.linalg.norm(uv_p[valid] - uv_d[valid], axis=1)
            errors.extend(diff.tolist())

    return float(np.mean(errors)) if errors else float("nan")

## PCK
def pck(pred: np.ndarray,
        gt: np.ndarray,
        thresholds: list[float] = [50.0, 100.0]) -> dict[float, float]:
    dists = np.linalg.norm(pred - gt, axis=-1) * 1000.0 
    return {delta: float((dists < delta).mean()) for delta in thresholds}

## full performance report helper
def report_metrics(label: str,
                   pred: np.ndarray,
                   gt: np.ndarray,
                   detections: np.ndarray,
                   projections: np.ndarray,
                   bones: list[tuple[int, int]],
                   pck_thresholds: list[float] = [50.0, 100.0]) -> dict:
    
    pck_vals = pck(pred, gt, pck_thresholds)

    m = {
        "MPJPE (mm)" : mpjpe(pred, gt),
        "PA-MPJPE (mm)" : pa_mpjpe(pred, gt),
        "Bone Length Std (mm)" : bone_length_std(pred, bones),
        "Mean Reproj (px)": mean_reproj_error(pred, detections, projections),
        **{f"PCK@{int(d)}mm" : v for d, v in pck_vals.items()},
    }
    print(f"Metrics — {label}")
    for k, v in m.items():
        print(f"  {k:<25} {v:.4f}")
    return m