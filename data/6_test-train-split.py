import pickle
from pathlib import Path
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sequence import Sample, Sequence

BASE_ROOT = Path("/gws/nopw/j04/iecdt/cheetah")
CSV = Path(__file__).parent / "sequences.csv"
SEED = 12345

    
class CheetahSamplesDataset(Dataset):
    def __init__(self, pkl_paths: List[Path], preload: bool = True):
        self.sequences = []
        if preload:
            for p in pkl_paths:
                if p.exists():
                    with open(p, "rb") as f:
                        seqs = pickle.load(f)  # list of Sample objects (a sequence) or dicts
                    # store sequence-level containers (each item is a sequence: list of samples)
                    self.sequences.append(seqs)
        else:
            raise NotImplementedError("Lazy loading not implemented in this snippet")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SamplesFromSequences(Dataset):
    """Flatten a sequence-level Subset (or iterable of sequences) into a sample-level Dataset."""
    def __init__(self, seq_subset):
        self.samples = []
        # handle torch.utils.data.Subset produced by random_split
        if isinstance(seq_subset, torch.utils.data.Subset):
            base = seq_subset.dataset
            indices = getattr(seq_subset, "indices", None)
            if indices is None:
                # fallback: iterate over subset
                for seq in seq_subset:
                    if isinstance(seq, list):
                        self.samples.extend(seq)
                    else:
                        self.samples.append(seq)
            else:
                for i in indices:
                    seq = base[i]
                    if isinstance(seq, list):
                        self.samples.extend(seq)
                    else:
                        self.samples.append(seq)
        else:
            # assume iterable of sequences
            for seq in seq_subset:
                if isinstance(seq, list):
                    self.samples.extend(seq)
                else:
                    self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_loaders(batch_size=32, val_frac=0.15, test_frac=0.2, num_workers=4):
    pkl_list = make_pkl_list(CSV, BASE_ROOT)
    dataset = CheetahSamplesDataset(pkl_list, preload=True)

    n = len(dataset)  # number of sequences
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    print(n_test, n_val, n_train)

    torch.manual_seed(SEED)

    # split sequences first
    train_seq_ds, val_seq_ds, test_seq_ds = random_split(dataset, [n_train, n_val, n_test])

    # then explode each split into sample-level datasets
    train_ds = SamplesFromSequences(train_seq_ds)
    val_ds = SamplesFromSequences(val_seq_ds)
    test_ds = SamplesFromSequences(test_seq_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def make_pkl_list(csv_path: Path, base_root: Path) -> List[Path]:
    df = pd.read_csv(csv_path)
    paths = [base_root / p / "sequence.pkl" for p in df["sequence_path"].astype(str).tolist()]
    return [p for p in paths if p.exists()]


    #     def __len__(self):
    #     return len(self.samples)

    # def __getitem__(self, idx):
    #     s = self.samples[idx]
    #     # adapt depending on Sample structure; here we return tensors
    #     x = torch.from_numpy(s.detections_2d)       # (C, J, 3)
    #     y = torch.from_numpy(s.ground_truth_3d)     # (J, 3)
    #     return x.float(), y.float(), s.frame_idx