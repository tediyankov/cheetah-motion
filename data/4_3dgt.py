import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# config --------------------------------------------------------------------------

BASE_DIR        = Path("/gws/nopw/j04/iecdt/cheetah/2017_08_29/bottom/phantom/flick2")
FTE_PICKLE      = BASE_DIR / "fte_pw" / "fte.pickle"
SCENE_JSON      = Path("/gws/nopw/j04/iecdt/cheetah/2017_08_29/bottom/extrinsic_calib/4_cam_scene_sba.json")
FILTERED_DIR    = BASE_DIR / "filtered_2D"
OUT_DIR         = BASE_DIR / "reprojection_verification"
VIDEO_DIR       = BASE_DIR 
VIDEO_ALPHA     = 0.35

N_SAMPLE_FRAMES = 10
RANDOM_SEED     = 42

# FTE joint order from their notebook
FTE_JOINT_NAMES = [
    "l_eye",           
    "r_eye",           
    "nose",            
    "neck_base",       
    "spine",           
    "tail_base",       
    "tail1", # (tail_mid)
    "tail2", # (tail_tip)
    "l_shoulder",  
    "l_front_knee",  
    "l_front_ankle",  
    "r_shoulder",     
    "r_front_knee",  
    "r_front_ankle",  
    "l_hip",          
    "l_back_knee",   
    "l_back_ankle",   
    "r_hip",          
    "r_back_knee",   
    "r_back_ankle",   
]

assert len(FTE_JOINT_NAMES) == 20, "Must have exactly 20 joint names to match FTE"

## scene calibration ------------------------------------------------------------

def load_scene(scene_json_path: Path) -> list[dict]:
    with open(scene_json_path) as f:
        data = json.load(f)

    cameras = []
    for cam in data["cameras"]:
        K = np.array(cam["k"], dtype=np.float64) # (3,3)
        D = np.array(cam["d"], dtype=np.float64) # (4,1)
        R = np.array(cam["r"], dtype=np.float64) # (3,3)
        t = np.array(cam["t"], dtype=np.float64) # (3,1)
        P = K @ np.hstack([R, t]) # (3,4)
        cameras.append({"K": K, "D": D, "R": R, "t": t, "P": P})

    return cameras

## projecting 3D points into a camera (pinhole, no distortion) ---------------------

def project_points(pts_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Project (J, 3) 3D points using (3, 4) projection matrix P.
    Returns (J, 2) pixel coordinates.
    """
    n = pts_3d.shape[0]
    pts_h = np.hstack([pts_3d, np.ones((n, 1))]) # (J, 4)
    proj = (P @ pts_h.T).T # (J, 3)
    px = proj[:, 0] / proj[:, 2]
    py = proj[:, 1] / proj[:, 2]
    return np.stack([px, py], axis=1) # (J, 2)


# main ------------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # loading FTE 3D positions
    with open(FTE_PICKLE, "rb") as f:
        fte = pickle.load(f)

    positions = np.array(fte["positions"]) # (200, 20, 3)
    start_frame = fte["start_frame"] # 13
    n_fte_frames, n_joints, _ = positions.shape
    print(f"FTE positions: {positions.shape}  start_frame={start_frame}")
    
    if n_joints != 20:
        print(f"WARNING: Expected 20 joints but got {n_joints}. Check FTE_JOINT_NAMES mapping!")

    # loading scene
    cameras = load_scene(SCENE_JSON)
    n_cameras = len(cameras)
    print(f"Loaded {n_cameras} cameras.")

    # loading filtered 2D DLC files (one per camera, sorted)
    h5_files = sorted(FILTERED_DIR.glob("cam*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No cam*.h5 files in {FILTERED_DIR}")
    assert len(h5_files) == n_cameras, \
        f"Expected {n_cameras} .h5 files, found {len(h5_files)}"

    dfs = [pd.read_hdf(h, key="df_with_missing") for h in h5_files]
    scorer = dfs[0].columns.get_level_values("scorer").unique()[0]
    
    # checking which DLC bodyparts are available
    dlc_bodyparts = dfs[0].columns.get_level_values('bodyparts').unique().tolist()
    print(f"\nDLC bodyparts available ({len(dlc_bodyparts)} total):")
    print(dlc_bodyparts)
    print(f"\nFTE joint names (20 total):")
    print(FTE_JOINT_NAMES)
    
    # checking that all FTE joints exist in DLC
    missing = [bp for bp in FTE_JOINT_NAMES if bp not in dlc_bodyparts]
    if missing:
        print(f"\nWARNING: These FTE joints are missing from DLC data: {missing}")

    # sampling frames (FTE-relative indices)
    rng = np.random.default_rng(RANDOM_SEED)
    fte_frame_idxs = rng.choice(n_fte_frames, size=N_SAMPLE_FRAMES, replace=False)
    fte_frame_idxs = np.sort(fte_frame_idxs)

    print(f"\nSampled FTE frame indices: {fte_frame_idxs}")

    for fte_idx in fte_frame_idxs:
        dlc_row = start_frame + fte_idx  # corresponding DLC row index

        pts_3d = positions[fte_idx] # (20, 3)

        fig, axes = plt.subplots(1, n_cameras, figsize=(6 * n_cameras, 6))
        fig.suptitle(f"Reprojection  |  FTE frame {fte_idx}  (DLC row {dlc_row})", fontsize=13)

        for cam_i, (ax, cam) in enumerate(zip(axes, cameras)):
            P = cam["P"]

            # reprojecting FTE 3D points
            proj_2d = project_points(pts_3d, P) # (20, 2)

            # loading 2D DLC detections for this camera at this frame
            df = dfs[cam_i]
            cam_num = int(h5_files[cam_i].name[3]) # filename cam3... → 3

            # plotting reprojected points
            ax.scatter(proj_2d[:, 0], proj_2d[:, 1],
                       c="red", s=40, marker="x", linewidths=1.5,
                       label="FTE reprojection", zorder=3)

            # annotating joint names on reprojected points
            for j, name in enumerate(FTE_JOINT_NAMES):
                ax.annotate(name, (proj_2d[j, 0], proj_2d[j, 1]),
                            fontsize=4, color="red", alpha=0.7)

            # plotting DLC 2D detections
            for j, bp in enumerate(FTE_JOINT_NAMES):
                x_col = (scorer, bp, "x")
                y_col = (scorer, bp, "y")
                if x_col not in df.columns:
                    continue
                x_val = df.at[dlc_row, x_col]
                y_val = df.at[dlc_row, y_col]
                if not np.isnan(x_val):
                    ax.scatter(x_val, y_val,
                            c="blue", s=20, marker="o", alpha=0.7,
                            zorder=2)
                    # annotating blue DLC points
                    ax.annotate(bp, (x_val, y_val),
                                fontsize=4, color="blue", alpha=0.7,
                                xytext=(3, 3), textcoords='offset points')

            # collecting all plotted points for auto-zoom
            all_u = list(proj_2d[:, 0])
            all_v = list(proj_2d[:, 1])
            for bp in FTE_JOINT_NAMES:
                x_col = (scorer, bp, "x")
                y_col = (scorer, bp, "y")
                if x_col in df.columns:
                    x_val = df.at[dlc_row, x_col]
                    y_val = df.at[dlc_row, y_col]
                    if not np.isnan(x_val):
                        all_u.append(x_val)
                        all_v.append(y_val)

            if len(all_u) > 0:
                pad = 100  # pixels of padding around the points
                ax.set_xlim(np.min(all_u) - pad, np.max(all_u) + pad)
                ax.set_ylim(np.max(all_v) + pad, np.min(all_v) - pad)  
            ax.set_aspect("equal")
            ax.set_title(f"cam{cam_num}", fontsize=11)
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")

        # legend
        red_patch  = mpatches.Patch(color="red",  label="FTE reprojection")
        blue_patch = mpatches.Patch(color="blue", label="DLC detection (filtered)")
        axes[-1].legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=8)

        plt.tight_layout()
        out_path = OUT_DIR / f"frame_{dlc_row:04d}_fte{fte_idx:04d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved → {out_path.name}")

    print("\nDone.")
    print("\nNote: FTE excludes paws (r_front_paw, l_front_paw, r_back_paw, l_back_paw)")
    print("      due to grass occlusion, as stated in the paper (Section III-C)")


if __name__ == "__main__":
    main()