import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from data.sequence import Sample, Sequence
   

# main
def main():
    # # read all the sequence paths from 'sequences.csv'
    # df = pd.read_csv("sequences.csv")
    # sequences = df["sequence_path"].tolist()

    # for seq in sequences:
    #     full_path = "/gws/nopw/j04/iecdt/cheetah/" + seq
    #     print(Path(full_path))

    #     sequence = Sequence(path = full_path)
    #     sequence.generate_samples()

    full_path = Path("/gws/nopw/j04/iecdt/cheetah/2017_08_29/bottom/phantom/flick2")
    sequence = Sequence(path = full_path)
    sequence.generate_samples()


if __name__ == "__main__":
    main()
    