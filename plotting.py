import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import numpy as np

def ppm_to_ic(ppm):
    ppm = np.clip(ppm, 1e-6, 1.0)
    entropy = -np.sum(ppm * np.log2(ppm), axis=1)
    ic = 2.0 - entropy

    return ppm * ic[:, None]


def plot_bonded_sequence_logo(ppm, title=None):
    columns = ["A", "C", "G", "T"]
    df = pd.DataFrame(ppm_to_ic(ppm), columns=columns)
    fig, ax = plt.subplots(figsize=(14, 3))

    logo = logomaker.Logo(df, ax=ax)
    # print("Maximum stack height:", ppm_to_ic(ppm).sum(axis=1).max())

    ax.set_xlabel("DNA Position")

    ax.set_ylabel("bits")

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()


def plot_ppms(ppm_fwd, ppm_rev, pdb_id):
    plot_bonded_sequence_logo(
        ppm_fwd, 
        title=f"{pdb_id} fwd"
    )
    plot_bonded_sequence_logo(
        ppm_rev,
        title=f"{pdb_id} rev",
    )
    plt.show()
