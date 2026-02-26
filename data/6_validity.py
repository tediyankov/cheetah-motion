import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

def _project(X_j: np.ndarray, P: np.ndarray) -> np.ndarray:
    # this takes 3D points X_J (J,3) and proj matrix (3,4) and projects into image space (J,2)
    # helper for when we check reproj error of DLT 
    X_h = np.concatenate([X_j, np.ones((X_j.shape[0], 1))], axis=1)
    uvw = (P @ X_h.T).T 
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv

def _load_sequence(seq_path: Path):
    # helpder to load seq pickle to give a list of Samples (ie a Sequence)
    pkl_path = seq_path / "sequence.pkl"
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    return samples

def _stack_detections(samples):
    # helper for building dense arrays for detections (T,J,C,3) and projj (T,C,3,4)
    T = len(samples)
    C, J, _ = samples[0].detections_2d.shape
    det = np.zeros((T, C, J, 3), dtype=np.float64)
    proj = np.zeros((T, C, 3, 4), dtype=np.float64)
    for t, s in enumerate(samples):
        det[t] = s.detections_2d
        proj[t] = s.camera_projections
    return det, proj

def compute_sequence_stats(seq_path: Path, conf_thresh: float = 0.3) -> dict:
    # loading detections and projections
    samples = _load_sequence(seq_path)
    detections, projections = _stack_detections(samples)
    T, C, J, _ = detections.shape

    # % of missing cameras ie frames where all proj are 0
    empty_cam = np.all(projections == 0, axis=(2, 3)) # (T, C)
    pct_missing_cams = empty_cam.mean()

    # valid 2D detections (if uv are finite and conf >= threshold)
    uv = detections[..., :2]
    conf = detections[..., 2]
    valid_uv = np.isfinite(uv).all(axis=-1)
    valid_conf = conf >= conf_thresh
    valid_det = valid_uv & valid_conf # (T, C, J)

    # how many cameras see each joint per frame (for DLT to triangulate a joint it needs at least 2 views)
    valid_views = valid_det.sum(axis=1) # (T, J)
    mean_valid_views = valid_views.mean()
    pct_joints_2plus = (valid_views >= 2).mean()

    # detection quality
    pct_nan_uv = (~valid_uv).mean()
    median_conf = np.nanmedian(conf)

    # DLT availability/quality
    dlt_path = seq_path / "dlt" / "norm_dlt.pkl"
    has_dlt = dlt_path.exists()
    dlt_pct_nan = np.nan
    dlt_mean_reproj = np.nan

    if has_dlt:
        with open(dlt_path, "rb") as f:
            X_dlt = np.array(pickle.load(f), dtype=np.float64)  # (T,J,3)
        dlt_nan_mask = ~np.isfinite(X_dlt).all(axis=-1)  # (T,J)
        dlt_pct_nan = dlt_nan_mask.mean()

        # reprojection error (px) using valid detections only
        errs = []
        for t in range(T):
            for c in range(C):
                P = projections[t, c]
                if np.all(P == 0):
                    continue
                v = valid_det[t, c]
                if not v.any():
                    continue
                uv_p = _project(X_dlt[t, v], P)
                uv_d = uv[t, c, v]
                diff = np.linalg.norm(uv_p - uv_d, axis=1)
                errs.extend(diff.tolist())
        dlt_mean_reproj = float(np.mean(errs)) if errs else np.nan

    return {
        "sequence": str(seq_path),
        "frames": T,
        "cams": C,
        "joints": J,
        "pct_missing_cams": float(pct_missing_cams),
        "mean_valid_views_per_joint": float(mean_valid_views),
        "pct_joints_with_2plus_views": float(pct_joints_2plus),
        "pct_nan_uv": float(pct_nan_uv),
        "median_conf": float(median_conf),
        "has_dlt": bool(has_dlt),
        "dlt_pct_nan_joints": float(dlt_pct_nan),
        "dlt_mean_reproj_px": float(dlt_mean_reproj),
    }

# helper for scoring rows for stable DLT and opt
def _score_row(r: pd.Series) -> float:
    if pd.isna(r.get("error")) is False:
        return np.nan

    # normalize to [0,1] then weight
    score = 0.0
    w = 0.0

    def add(val, lo, hi, weight):
        nonlocal score, w
        if pd.isna(val):
            return
        v = (val - lo) / (hi - lo)
        v = float(np.clip(v, 0.0, 1.0))
        score += v * weight
        w += weight

    # higher is better
    add(r["mean_valid_views_per_joint"], 2.0, 4.0, 2.0)
    add(r["pct_joints_with_2plus_views"], 0.6, 0.95, 2.0)
    add(r["median_conf"], 0.4, 0.9, 1.5)

    # lower is better
    add(1.0 - r["pct_missing_cams"], 0.7, 1.0, 1.0)
    add(1.0 - r["pct_nan_uv"], 0.7, 1.0, 1.0)

    if r.get("has_dlt", False):
        add(1.0 - r["dlt_pct_nan_joints"], 0.6, 1.0, 1.0)
        add(1.0 - (r["dlt_mean_reproj_px"] / 10.0), 0.5, 1.0, 1.5)

    if w == 0:
        return np.nan

    # map to 1–10
    return float(1 + 9 * (score / w))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences_csv", type=Path, default=Path("sequences.csv"))
    parser.add_argument("--root", type=Path, default=Path("/gws/nopw/j04/iecdt/cheetah"))
    parser.add_argument("--out_csv", type=Path, default=Path("sequence_quality_report.csv"))
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    df = pd.read_csv(args.sequences_csv)
    seqs = df["sequence_path"].tolist()

    rows = []
    for s in seqs:
        seq_path = args.root / s
        try:
            rows.append(compute_sequence_stats(seq_path, conf_thresh=args.conf))
        except Exception as e:
            rows.append({
                "sequence": str(seq_path),
                "error": str(e),
            })

    out_df = pd.DataFrame(rows)
    # quality score 1–10
    out_df["quality_score_1_10"] = out_df.apply(_score_row, axis=1)

    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote report → {args.out_csv}")

    # quick summary
    if "quality_score_1_10" in out_df.columns:
        top = out_df.sort_values("quality_score_1_10", ascending=False).head(5)
        print("\ntop 5 sequences:")
        print(top[["sequence", "quality_score_1_10"]].to_string(index=False))
    
    print(f"Wrote report → {args.out_csv}")


if __name__ == "__main__":
    main()

# TODO: re-run DLT script on all sequences to get norm_dlt.pkl

# now need to pick a seq that has 
# - low % missing cameras
# - high mean_valid_views_per_joint (over 3 at least)
# - high pct_joints_with_2plus_views (over 0.8 ideally)
# - low pct_nan_uv (near 0)
# - high median_conf (over or at 0.7 ideally)
# - has DLT obviously
# - low dlt_pct_nan_joints (less than 0.2)
# - low dlt_mean_reproj_px (under 5px ideally)