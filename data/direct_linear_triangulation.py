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

    # normalize 1D inputs to (N,2)
    if us_vs.ndim == 1 and us_vs.size in (2,3):
        us_vs = us_vs.reshape(1, -1)

    if us_vs.ndim != 2 or us_vs.shape[1] < 2:
        return None
        raise ValueError(f"us_vs must be shape (N,2) or (N,3); got shape {us_vs.shape}")

    # keep only first two columns (u,v)
    us_vs = us_vs_cs[:, :2]

    # Coerce Ps to list and check length
    Ps = list(Ps)
    if len(Ps) != len(us_vs):
        # number of cameras should equal number of detections
        # note: at this stage this may still incude invalid "nan" detections
        return None
        raise ValueError(f"Number of projection matrices ({len(Ps)}) "
                         f"must equal number of observations ({len(us_vs)})")

    A_rows = []
    for i, ((u, v), P) in enumerate(zip(us_vs, Ps)):
        P = np.asarray(P)
        if P.shape != (3,4):
            return None
            raise ValueError(f"Projection matrix at index {i} has shape {P.shape}, expected (3,4)")
        p1 = P[0, :]
        p2 = P[1, :]
        p3 = P[2, :]

        # if u or v is nan skip because it means this view isn't valid 
        # us and vs will be nan when the confidence was <0.3
        if np.isnan(u) or np.isnan(v):
            continue

        # otherwise we are safe to trust these coords as valid detections
        row1 = u * p3 - p1
        row2 = v * p3 - p2

        A_rows.append(row1)
        A_rows.append(row2)

        # skip rows containing NaN or Inf
        if np.all(np.isfinite(row1)):
            A_rows.append(row1)
        if np.all(np.isfinite(row2)):
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

        write_path = write_dir / "dlt.pkl"

        # write the results to a pickle file at dlt/dlt.pickle
        print(f"Writing dlt results to {write_dir}")
        pickle.dump(results, open(write_path, "wb"))

if __name__ == "__main__":
    main()
    