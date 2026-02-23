import pandas as pd
from pathlib import Path
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import pickle


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



@dataclass
class Sample:
    # TODO: include the path to the data so we know where the frame came from in the same way that frame_idx was included
    """
    Represents one frame's data for a particular camera inside a Sequence.
    detections_2D: np.ndarray, shape (C=6, J, 3) where C isnumber of cameras
            J is number of bodyparts, and the last dim is (u, v, likelihood).
    frame_idx: int (0-based row index in the dataframe)
    """
    # these are the crucial parts as specified in data processing instructions
    detections_2d: np.ndarray 
    camera_projections: any # establish what type this should be, list of
    ground_truth_3d: np.ndarray

    # useful information
    frame_idx: int # frame_id from which this comes
#     detections_2d_file: Path # file path to filtered h5 2d data array file
#     ground_truth_3d_file: Path # file path to 3d ground truth

class Sequence:
    def __init__(self, path: Path):
        self.path = path
        self.detections_2d_dir = Path(self.path + "/filtered_2D")
        self.ground_truth_3d_file = Path(self.path + "/fte_pw/fte.pickle")
        self.samples: List[Sample] = []
        self.sequence_path = Path(self.path + "/sequence.pkl")
        self.undistored_dir = Path(self.path + "/undistorted_2D")
        self.camera_projections = np.zeros((6, 3, 4))

    def load_proj_matrices(self):

        for file in self.undistored_dir.glob("cam*.npz"):
            match = re.search(r'cam(\d+)', file.name)

            cam_id = int(match.group(1)) 
            cam_idx = cam_id - 1 # convert to 0-based index

            proj = np.load(file)

            self.camera_projections[cam_idx, :, :] = proj["P"]

    def load_3d_ground_truth(self):
        # loading FTE 3D positions
        if not self.ground_truth_3d_file.exists():
            raise FileNotFoundError(f"3D ground-truth file not found: {self.ground_truth_3d_file}")

        with open(self.ground_truth_3d_file, "rb") as f:
            fte = pickle.load(f)
        
        # pickle file is a dictionary with keys dict_keys(['positions', 'x', 'dx', 'ddx', 'start_frame'])
        # the positions is a nested list that can be converted to an array of dimension (F, J, 3)
        # where F is number of frames, J is number of joints (25) and 3 are the 3D coordinates
        fte_df = np.array(fte["positions"], dtype=np.float32)
        start_frame = fte["start_frame"]

        return fte_df, start_frame

   
    def load_2d_detections(self):

        cam_data: Dict[int, np.ndarray] = {}

        # loop through the filtered directory
        for file in self.detections_2d_dir.glob("cam*.h5"):

            # assumption: all files will begin "camN" where N is the camera number
            # extract camera number from file name
            match = re.search(r'cam(\d+)', file.name)
            if not match:
                print(f"Warning: Could not parse camera ID from {file.name}")
                continue

            cam_id = int(match.group(1)) 
            cam_idx = cam_id - 1 # convert to 0-based index

            print(f"Loading filtered detections from {file} (camera ID {cam_id})")

            df = pd.read_hdf(file, key="df_with_missing")

            # drop the top level, leaving two levels (bodyparts, coords)
            df.columns = df.columns.droplevel(0)

            # can we drop the paw joints here so we have same number as ground truth
            # also need to ensure the ordering of joints is the same
            df = df[FTE_JOINT_NAMES]

            x_df = df.xs("x", axis=1, level="coords")
            y_df = df.xs("y", axis=1, level="coords")
            c_df = df.xs("likelihood", axis=1, level="coords")

            x_np = x_df.to_numpy(dtype=np.float32)  
            y_np = y_df.to_numpy(dtype=np.float32)
            c_np = c_df.to_numpy(dtype=np.float32)

            stacked = np.stack([x_np, y_np, c_np], axis=2)  # (F, J, 3)

            cam_data[cam_idx] = stacked
            
        return cam_data

        
    def generate_samples(self):
        """
        Creates self.samples using:
          - the ground-truth timeline); for each GT frame, find the corresponding
            row in each camera's DataFrame, build a (C, J, 3) tensor and attach GT Jx3.
        Alignment:
          - FTE provides a 'start_frame' value, global_frame = start_frame + gt_idx
        """
        # ensure we start with a clean slate 
        self.samples = []

        self.load_proj_matrices()
        fte_arr, start_frame = self.load_3d_ground_truth()

        # number of frames, number of joints, coordinate dimension
        F = fte_arr.shape[0]
        J = fte_arr.shape[1]
        C = 6

        cam_dfs = self.load_2d_detections()

        for i in range(F):

            global_frame = start_frame + i

            # initialize base tensor, size ()
            tensor = np.zeros((C, J, 3), dtype=np.float32)

            for cam_idx in range(C):

                if cam_idx not in cam_dfs:
                    continue
                
                cam_df = cam_dfs[cam_idx]

                # for each camera, get the positions for that frame_id
                # what to do if camera doesn't have this frame?

                row = global_frame - 1 # DLC dataframe is zero indexed, is fte pickle?
                tensor[cam_idx, :, :] = cam_df[row]

        
            sample = Sample(
                detections_2d=tensor, 
                camera_projections=self.camera_projections, # implement this (C, projmatrix.shape)
                ground_truth_3d = fte_arr[i],
                frame_idx = global_frame,
            )

            self.samples.append(sample)

        self.write_to_pickle()
        
        return
    
    def write_to_pickle(self):
        print(f"Writing constructed sequence to {self.sequence_path}")
        sequence = self.samples
        pickle.dump(sequence, open(self.sequence_path, "wb"))