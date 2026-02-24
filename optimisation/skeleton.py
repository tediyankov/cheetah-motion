
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sequence import FTE_JOINT_NAMES

FTE_PICKLE = Path("/gws/nopw/j04/iecdt/cheetah/2017_12_12/bottom/big_girl/flick2/fte_pw/fte.pickle")
OUT_DIR = Path("/gws/nopw/j04/iecdt/tyankov/cheetah-motion/optimisation")

J = FTE_JOINT_NAMES
IDX = {name: i for i, name in enumerate(FTE_JOINT_NAMES)}

BONES_WITH_LENGTHS = [
    # head region 
    ("neck_base", "l_eye", 0.28), # approximate via head offset
    ("neck_base", "r_eye", 0.28),
    ("neck_base", "nose", 0.28),

    # spine
    ("neck_base", "spine", 0.37),
    ("spine", "tail_base", 0.37),
    ("tail_base", "tail1",0.28), # tail1 = tail_mid
    ("tail1", "tail2",0.36), # tail2 = tail_tip

    # left forelimp
    ("neck_base", "l_shoulder", np.linalg.norm([-0.04,  0.08, -0.10])),
    ("l_shoulder", "l_front_knee", 0.24),
    ("l_front_knee", "l_front_ankle", 0.28),

    # right forelimb 
    ("neck_base","r_shoulder", np.linalg.norm([-0.04, -0.08, -0.10])),
    ("r_shoulder",   "r_front_knee", 0.24),
    ("r_front_knee", "r_front_ankle", 0.28),

    # left hindlimb
    ("tail_base", "l_hip", np.linalg.norm([0.12,  0.08, -0.06])),
    ("l_hip", "l_back_knee", 0.32),
    ("l_back_knee", "l_back_ankle",0.25),

    # right hindlimb 
    ("tail_base","r_hip", np.linalg.norm([0.12, -0.08, -0.06])),
    ("r_hip", "r_back_knee", 0.32),
    ("r_back_knee","r_back_ankle", 0.25),
]

## converting to index based tuples
BONES = [(IDX[p], IDX[c]) for p, c, _ in BONES_WITH_LENGTHS]
BONE_LENGTHS_M  = np.array([l for _, _, l in BONES_WITH_LENGTHS]) 
BONE_LENGTHS_MM = BONE_LENGTHS_M * 1000.0  

## loading FTE positions to visualise skeleton on mean pose
with open (FTE_PICKLE, 'rb') as f:
    fte = pkl.load (f)

positions = np.array (fte['positions'], dtype=np.float64)
mean_pos = np.nanmean(positions, axis = 0)

## obs vs paper bone lengths
print(f"\n{'Bone':<35} {'Paper (mm)':>10} {'FTE mean (mm)':>14} {'Diff (mm)':>10}")
print("-" * 73)
for (i, k), l_paper in zip(BONES, BONE_LENGTHS_MM):
    name_i = FTE_JOINT_NAMES[i]
    name_k = FTE_JOINT_NAMES[k]
    lengths = np.linalg.norm(positions[:, i, :] - positions[:, k, :], axis=-1)
    l_fte = np.nanmean(lengths)
    diff = l_fte - l_paper
    print(f"{name_i:<18} → {name_k:<18} {l_paper:>10.1f} {l_fte:>14.1f} {diff:>10.1f}")

## viz
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection="3d")

ax.scatter(mean_pos[:, 0], mean_pos[:, 1], mean_pos[:, 2],
           c="steelblue", s=60, zorder=5)

for j, name in enumerate(FTE_JOINT_NAMES):
    ax.text(mean_pos[j, 0], mean_pos[j, 1], mean_pos[j, 2],
            f" {j}:{name}", fontsize=6, color="black")

for i, k in BONES:
    xs = [mean_pos[i, 0], mean_pos[k, 0]]
    ys = [mean_pos[i, 1], mean_pos[k, 1]]
    zs = [mean_pos[i, 2], mean_pos[k, 2]]
    ax.plot(xs, ys, zs, c="tomato", linewidth=2)

ax.set_title("Cheetah skeleton from paper", fontsize=12)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

out_path = OUT_DIR / "skeleton.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved skeleton visualisation → {out_path}")


