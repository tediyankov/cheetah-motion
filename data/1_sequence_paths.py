from pathlib import Path
from typing import List
import re

def is_date_folder(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"\d{4}_\d{2}_\d{2}", path.name)

def find_all_sequences(dataset_root: str) -> List[str]:
    """
    Return all sequence paths relative to dataset_root.
    Output format:
        YYYY_MM_DD/top|bottom/<cheeetah_name>/<manoeuvre_type>
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"{dataset_root} does not exist")

    sequences = set()

    # Only scan valid date folders
    for date_dir in sorted(root.iterdir()):
        if not is_date_folder(date_dir):
            continue

        # Find all fte_pw directories inside this date
        for fte_dir in date_dir.rglob("fte_pw"):
            seq_dir = fte_dir.parent

            # Return path INCLUDING date (relative to dataset root)
            rel_path = seq_dir.relative_to(root)
            sequences.add(rel_path.as_posix())

    return sorted(sequences)


def save_to_csv(paths: List[str], outpath: str) -> None:
    import csv
    with open(outpath, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence_path"])
        for p in paths:
            w.writerow([p])


if __name__ == "__main__":
    dataset_root = "/gws/nopw/j04/iecdt/cheetah"   # <-- change this to your dataset root
    sequences = find_all_sequences(dataset_root)
    print(f"Found {len(sequences)} sequences:\n")
    for s in sequences:
        print(s)

    save_to_csv(sequences, "sequences.csv")