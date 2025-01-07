import random

import numpy as np


# aa composition: https://web.expasy.org/docs/relnotes/relstat.html
POPULATION = ["A", "Q", "L", "S", "R", "E", "K", "T", "N", "G", "M", "W", "D", "H", "F", "Y", "C", "I", "P", "V"]
WEIGHTS = [.0825, .0393, .0965, .0664, .0553, .0672, .0580, .0535, .0406, .0707, .0241, .0110, .0546, .0227, .0386, .0292, .0138, .0591, .0474, .0686]

def pre_pad_protein_sequences(sequences: np.ndarray[np.str_], maxlen: int) -> np.ndarray[np.str_]:
    padded_sequences = []
    for seq in sequences:
        k = maxlen - len(seq)
        if k > 0:
            aa_seq = "".join(random.choices(POPULATION, WEIGHTS, k=k))
            padded_sequences.append(aa_seq + seq)
        else:
            padded_sequences.append(seq)
    return np.asarray(padded_sequences)

def post_pad_protein_sequences(sequences: np.ndarray[np.str_], maxlen: int) -> np.ndarray[np.str_]:
    padded_sequences = []
    for seq in sequences:
        k = maxlen - len(seq)
        if k > 0:
            aa_seq = "".join(random.choices(POPULATION, WEIGHTS, k=k))
            padded_sequences.append(seq + aa_seq)
        else:
            padded_sequences.append(seq)
    return np.asarray(padded_sequences)
