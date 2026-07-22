import numpy as np


def split_validation(dataset, indices, val_frac, seed):
    """
    Carve a class-stratified validation slice out of a training pool, so
    training progress can be checked against subjects the model never
    trains on. At least 1 subject per group is kept for training.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for idx in indices:
        groups.setdefault(dataset.subjects[idx]["group"], []).append(idx)

    val_indices = []
    for idxs in groups.values():
        n_val = min(max(1, round(len(idxs) * val_frac)), len(idxs) - 1)
        val_indices += rng.choice(idxs, size=n_val, replace=False).tolist()

    train_indices = [i for i in indices if i not in val_indices]
    return train_indices, val_indices
