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
                    # normalize sequence entries to simple tuples/lists so downstream
                    # datasets don't have to handle `Sample` instances directly
                    normalized = []

                    for sample in seqs:
                        # sample has detections_2d and ground_truth_3d
                        # we need to load these as a tuple
                        cleaned_tuple = (torch.from_numpy(sample.detections_2d), torch.from_numpy(sample.ground_truth_3d))
                        normalized.append(cleaned_tuple)

                    # store sequence-level containers (each item is a sequence: list of cleaned samples)
                    self.sequences.append(normalized)
        else:
            raise NotImplementedError("Lazy loading not implemented in this snippet")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SamplesFromSequences(Dataset):
    """Flatten a sequence-level Subset (or iterable of sequences) into a sample-level Dataset.
    Each item returned is a tuple `(detections_2d_tensor, ground_truth_3d_tensor)` where tensors
    are `torch.float` and suitable for batching in a DataLoader.
    """
    def __init__(self, seq_subset):
        self.samples = []

        # def _to_tensor(x):
        #     if isinstance(x, torch.Tensor):
        #         return x.float()
        #     import numpy as _np
        #     return torch.from_numpy(_np.asarray(x)).float()

        def _append_sample(s):
            self.samples.append((s[0], s[1]))

        # seq_subset (including torch.utils.data.Subset) is iterable; iterate and flatten
        # support both sequence containers (lists of samples) and single-sample entries
        for seq in seq_subset:
            if isinstance(seq, list):
                for s in seq:
                    _append_sample(s)
            else:
                _append_sample(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_data_loaders(batch_size=32, val_frac=0.15, test_frac=0.2, num_workers=4):
    """
    Returns train, val, test dataloaders.
    """
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
