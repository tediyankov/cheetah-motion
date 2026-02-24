import pandas as pd
from sequence import Sample, Sequence
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def remove_empty_cameras(sample: Sample) -> Sample:
    # the data processing step sets projection matrices to 0 for missing cameras
    # but keeps them in the tensor so we always have a (6,3,4) size and can infer
    # which camera id is there
    # this same convention is followed for all data

    # determine which are the "missing" cameras
    cam_projs = sample.camera_projections # shape (6, 3, 4)
    empty_mask = np.all(cam_projs == 0, axis=(1,2))
    keep_mask = ~empty_mask

    # remove the "missing" cameras and corresponding detections (which would just be zeroed)
    new_proj = cam_projs[keep_mask]
    new_dets = sample.detections_2d[keep_mask]

    # return the cleaned frame
    return Sample(
        detections_2d=new_dets,
        camera_projections=new_proj,
        ground_truth_3d=sample.ground_truth_3d,
        frame_idx=sample.frame_idx
    )

def normalise_points_2d (points): 
    # TODO: points might have NaNs so we need to deal with that first before getting the centroid etc
    points = np.asarray(points, dtype = float)

    # keep finite rows
    valid_mask = np.isfinite(points).all(axis=1)
    if not np.any(valid_mask):
        raise ValueError("normalise_points_2d: no finite points to normalise")

    valid_points = points[valid_mask]

    centroid = valid_points.mean (axis = 0)
    shifted = valid_points - centroid
    dist = np.linalg.norm (shifted, axis = 1)
    mean_dist = dist.mean()

    if mean_dist == 0 or not np.isfinite(mean_dist):
        raise ValueError("normalise_points_2d: invalid mean distance")

    scaling_factor = np.sqrt(2) / mean_dist

    T = np.array ([
        [scaling_factor, 0, -scaling_factor * centroid[0]],
        [0, scaling_factor, -scaling_factor * centroid[1]],
        [0,0,1]
    ])

    points_homog = np.column_stack((points, np.ones(points.shape[0])))
    out = (T @ points_homog.T).T[:, :2]

    # keep nans where input was invalid
    out[~valid_mask] = np.nan
    return {'coords': out, 'matrix': T}

def triangulate_joint_dlt(us_vs_cs, Ps): 
    """
    Triangulate a single 3D point from C image observations using DLT (SVD).
    us_vs_cs: (C, 3) array of C camera detections (u,v,c) where c is the confidence level
    Ps    : sequence of C camera projection matrices (each 3x4)
    returns: (X,Y,Z) 3-vector in non-homogeneous coords
    """
    us_vs_cs = np.asarray(us_vs_cs)
    # if the joint is not detected with sufficient confidence, do not include this camera view
    # note: we have already done this in the data processing step, and have set the (u,v) to nan whenever c<0.3

    # keep only first two columns (u,v)
    us_vs = us_vs_cs[:, :2]

    # TODO: keep only valid views
    valid_mask = np.isfinite(us_vs).all(axis=1)
    if np.sum(valid_mask) < 2:
        return None

    us_vs = us_vs[valid_mask]
    Ps = np.asarray(Ps)[valid_mask]

    # apply norm
    try:
        norm_dict = normalise_points_2d(us_vs) 
    except ValueError as e:
        print(f"normalise_points_2d failed: {e}")
        return None
    
    us_vs_norm = norm_dict['coords']
    T = norm_dict['matrix']

    # Coerce Ps to list and check length
    Ps = list(Ps)
    if len(Ps) != len(us_vs_norm):
        # number of cameras should equal number of detections
        # note: at this stage this may still incude invalid "nan" detections
        return None
        raise ValueError(f"Number of projection matrices ({len(Ps)}) "
                         f"must equal number of observations ({len(us_vs_norm)})")

    A_rows = []
    for i, ((u, v), P) in enumerate(zip(us_vs_norm, Ps)):
        P = np.asarray(P)
        # normalise projection matrices using the same T as the points
        P_norm = T @ P
        if P_norm.shape != (3,4):
            return None
            raise ValueError(f"Projection matrix at index {i} has shape {P_norm.shape}, expected (3,4)")
        p1 = P_norm[0, :]
        p2 = P_norm[1, :]
        p3 = P_norm[2, :]

        # if u or v is nan skip because it means this view isn't valid 
        # us and vs will be nan when the confidence was <0.3
        if np.isnan(u) or np.isnan(v):
            continue

        # otherwise we are safe to trust these coords as valid detections
        row1 = u * p3 - p1
        row2 = v * p3 - p2

        A_rows.append(row1)
        A_rows.append(row2)

    if len(A_rows) < 2:
        return None
        raise ValueError("Not enough valid (finite) observations to triangulate")

    A = np.vstack(A_rows)   # shape (M,4) where M = # valid detections
    # Solve A X = 0 via SVD
    U, s, Vt = np.linalg.svd(A)
    X_hom = Vt[-1, :]
    if np.isclose(X_hom[-1], 0):
        return None
        raise ValueError("Degenerate result: last homogeneous coordinate is zero (or extremely small).")
    X = X_hom / X_hom[-1]
    return X[:3].astype(float)

def triangulate_all_joints_per_sample(sample: Sample) -> np.ndarray:
    
    # clean the sample by removing the empty cameras
    clean_sample = remove_empty_cameras(sample)

    sample_result = np.zeros((20,3))

    for j in range(20):
        # need the (u,v) observations for the same 3D point across all cameras 
        xs_ys_cs = clean_sample.detections_2d[:, j, :]
        Ps = clean_sample.camera_projections
        
        coords = triangulate_joint_dlt(xs_ys_cs, Ps) 
        if coords is None:
            # if there is an error in joint triangulation, fill with nans
            # TODO(ueastwood) propogate errors up more appropriately
            sample_result[j] = np.nan
        else:      
            sample_result[j] = coords
    
    return sample_result


def triangulate_entire_sequence(sequence: List[Sample]) -> np.ndarray:
    """For an entire sequence (list of samples):
        - for each frame(note a frame is a sample):
            - feed in the frame to triangulate all joints 
            - append to results
        - final results will have shape (#FRAMES, 20, 3) where 20 = #joints and 3 is X,Y,Z coords
    """
    NUM_FRAMES = len(sequence)
    results = np.zeros((NUM_FRAMES, 20, 3))
    for i, sample in enumerate(sequence):
        # if there is an error in triangulate_all_joints just append nans to the results
        
        results[i] = triangulate_all_joints_per_sample(sample)

    return results


def main():
    # read all the sequence paths 
    df = pd.read_csv("sequences.csv")
    sequences = df["sequence_path"].tolist()

    for seq in sequences:
        full_path = "/gws/nopw/j04/iecdt/cheetah/" + seq

        with open(full_path + "/sequence.pkl", "rb") as f:
            test_seq = pickle.load(f)

        results = triangulate_entire_sequence(test_seq)
        
        write_dir = Path(full_path + "/dlt")
        write_dir.mkdir(parents=True, exist_ok=True)

        write_path = write_dir / "norm_dlt.pkl"

        # write the results to a pickle file at dlt/dlt.pickle
        print(f"Writing dlt results to {write_path}")
        pickle.dump(results, open(write_path, "wb"))

if __name__ == "__main__":
    main()
    