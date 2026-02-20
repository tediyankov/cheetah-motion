import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import re

## loading scene calibration --------------------------------------------------------

def load_scene(scene_json_path: Path) -> dict:
    with open(scene_json_path) as f:
        data = json.load(f)

    cameras = []
    for cam in data["cameras"]:
        K = np.array(cam["k"], dtype=np.float64)  
        D = np.array(cam["d"], dtype=np.float64)   
        R = np.array(cam["r"], dtype=np.float64)   
        t = np.array(cam["t"], dtype=np.float64)   
        P = K @ np.hstack([R, t])                  
        cameras.append({"K": K, "D": D, "R": R, "t": t, "P": P})

    return cameras


## undistorted points ---------------------------------------------------------------

def undistort_points(pts_xy: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    out = pts_xy.copy()
    valid = ~np.isnan(pts_xy).any(axis=1)

    if valid.any():
        pts_in  = pts_xy[valid].astype(np.float64).reshape(1, -1, 2)
        pts_out = cv2.fisheye.undistortPoints(pts_in, K, D, P=K)
        out[valid] = pts_out.reshape(-1, 2)

    return out


## main ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Undistort DLC 2D keypoints using fisheye camera model"
    )
    parser.add_argument(
        "--dlc-dir",
        type=Path,
        required=True,
        help="Path to directory containing DLC .h5 files (e.g., .../flick2/dlc)"
    )
    parser.add_argument(
        "--scene-json",
        type=Path,
        required=True,
        help="Path to scene calibration JSON file"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <dlc-dir>/../undistorted_2D)"
    )
    
    args = parser.parse_args()
    
    DLC_DIR = args.dlc_dir
    SCENE_JSON = args.scene_json
    OUT_DIR = args.out_dir if args.out_dir else DLC_DIR.parent / "undistorted_2D"
    
    # validate inputs
    if not DLC_DIR.exists():
        raise FileNotFoundError(f"DLC directory not found: {DLC_DIR}")
    if not SCENE_JSON.exists():
        raise FileNotFoundError(f"Scene JSON not found: {SCENE_JSON}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cameras = load_scene(SCENE_JSON)
    print(f"Loaded {len(cameras)} cameras from scene JSON.")

    h5_files = sorted(DLC_DIR.glob("cam*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No cam*.h5 files found in {DLC_DIR}")

    for scene_idx, h5_path in enumerate(h5_files):
        cam_num = int(h5_path.name[3])   # for logging only
        print(f"\nProcessing {h5_path.name}  (cam_num={cam_num}, scene_idx={scene_idx})")

        K = cameras[scene_idx]["K"]
        D = cameras[scene_idx]["D"]
        P = cameras[scene_idx]["P"]

        # derive camera name from h5 file (eg cam1DLC_ -> cam1)
        m = re.match(r"(cam\d+)", h5_path.stem)
        cam_name = m.group(1) if m else f"cam{scene_idx+1}"

        # save per-camera P matrix as .npz
        p_out = OUT_DIR / f"{cam_name}_P.npz"
        np.savez(p_out, P=P)
        print(f"Saved → {p_out}")

        df = pd.read_hdf(h5_path, key="df_with_missing")
        df_out = df.copy()

        scorer = df.columns.get_level_values("scorer").unique()[0]
        bodyparts = df.columns.get_level_values("bodyparts").unique()

        for bp in bodyparts:
            x_col = (scorer, bp, "x")
            y_col = (scorer, bp, "y")

            xy = df[[x_col, y_col]].values.astype(np.float64)  
            xy_undist = undistort_points(xy, K, D)                     

            df_out[x_col] = xy_undist[:, 0]
            df_out[y_col] = xy_undist[:, 1]

        out_path = OUT_DIR / h5_path.name
        df_out.to_hdf(str(out_path), key="df_with_missing", mode="w")
        print(f"Saved → {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()