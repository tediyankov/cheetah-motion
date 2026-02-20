
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# main ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filter DLC 2D keypoints by confidence threshold"
    )
    parser.add_argument(
        "--undistorted-dir",
        type=Path,
        required=True,
        help="Path to directory containing undistorted .h5 files"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <undistorted-dir>/../filtered_2D)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for filtering (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    UNDISTORTED_DIR = args.undistorted_dir
    OUT_DIR = args.out_dir if args.out_dir else UNDISTORTED_DIR.parent / "filtered_2D"
    CONF_THRESHOLD = args.conf_threshold
    
    # Validate inputs
    if not UNDISTORTED_DIR.exists():
        raise FileNotFoundError(f"Undistorted directory not found: {UNDISTORTED_DIR}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(UNDISTORTED_DIR.glob("cam*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No cam*.h5 files found in {UNDISTORTED_DIR}")

    for h5_path in h5_files:
        print(f"\nProcessing {h5_path.name}")

        df = pd.read_hdf(h5_path, key="df_with_missing")
        df_out = df.copy()

        scorer = df.columns.get_level_values("scorer").unique()[0]
        bodyparts = df.columns.get_level_values("bodyparts").unique()

        n_total = 0
        n_dropped = 0

        for bp in bodyparts:
            x_col = (scorer, bp, "x")
            y_col = (scorer, bp, "y")
            c_col = (scorer, bp, "likelihood")

            confidence = df[c_col].values   # (T,)
            low_conf = confidence < CONF_THRESHOLD

            n_total += len(confidence)
            n_dropped += low_conf.sum()

            # setting x and y to NaN where confidence is too low
            df_out.loc[low_conf, x_col] = np.nan
            df_out.loc[low_conf, y_col] = np.nan

        out_path = OUT_DIR / h5_path.name
        df_out.to_hdf(str(out_path), key="df_with_missing", mode="w")

        print(f"  Detections dropped : {n_dropped}/{n_total} ({100*n_dropped/n_total:.1f}%)")
        print(f"  Saved → {out_path}")

    # per-frame summary: for each frame and joint, how many cameras have valid detections
    print("\n--- Valid detection counts per frame (across cameras) ---")
    summarise_valid_views(OUT_DIR, CONF_THRESHOLD)

    print("\nDone.")


def summarise_valid_views(filtered_dir: Path, threshold: float) -> None:
    """
    For each frame and each joint, print how many cameras have valid (non-NaN) detections.
    Triangulation requires >= 2 valid views.
    """
    h5_files = sorted(filtered_dir.glob("cam*.h5"))

    # loading all cameras
    dfs = [pd.read_hdf(h, key="df_with_missing") for h in h5_files]
    scorer = dfs[0].columns.get_level_values("scorer").unique()[0]
    bodyparts = dfs[0].columns.get_level_values("bodyparts").unique()

    # align frame counts (truncate to shortest)
    frame_counts = [len(df) for df in dfs]
    n_frames = min(frame_counts)
    if len(set(frame_counts)) > 1:
        print(f"\nWARNING: Mismatched frame counts {frame_counts}. "
              f"Truncating all to {n_frames} frames for summary.")
    dfs = [df.iloc[:n_frames].copy() for df in dfs]

    # for each bodypart, counting how many cameras have valid detections per frame
    valid_counts = {}
    for bp in bodyparts:
        x_col = (scorer, bp, "x")
        valid = np.stack(
            [~np.isnan(df[x_col].values) for df in dfs],
            axis=1
        )
        valid_counts[bp] = valid.sum(axis=1)

    # summary table: per joint, fraction of frames with >= 2 valid views
    print(f"\n{'Bodypart':<25} {'>=2 views':>10} {'>=1 view':>10} {'0 views':>10}")
    print("-" * 57)
    for bp in bodyparts:
        counts = valid_counts[bp]
        gte2 = (counts >= 2).sum()
        gte1 = (counts >= 1).sum()
        zero = (counts == 0).sum()
        print(f"{bp:<25} {gte2:>9d}  {gte1:>9d}  {zero:>9d}")

    print(f"\nTotal frames: {n_frames}")
    print(f"Frames where ALL joints have >=2 views: {sum((valid_counts[bp] >= 2).all() for bp in bodyparts)}")


if __name__ == "__main__":
    main()
