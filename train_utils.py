import contextlib
import os
import sys

import numpy as np


class _Tee:
    """Writes to multiple streams at once -- see tee_stdout_to_file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout_to_file(log_path):
    """
    Duplicates everything written to stdout (every existing print(...,
    flush=True) call in the train_*.py scripts) to log_path as well, without
    needing to touch any of those individual print calls. Overwrites
    log_path each run, same as the checkpoint .pt files this project already
    overwrites on each run -- the log is "this training run's output," not
    an accumulating history. Creates log_path's parent directory (e.g.
    logs/) if it doesn't exist yet, so this works on a fresh checkout too.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout


CHECKPOINT_DIR = "saved_models"
SEEDS = [42, 43, 44]


def seed_checkpoint_name(checkpoint_name, seed):
    """Seed 42 uses the canonical unsuffixed checkpoint; seeds 43/44 use the
    `_seed{N}` suffix inserted before `.pt`, matching the convention the
    run_multiseed_{tuh,caueeg}.py torch.save monkeypatch writes at training
    time. Every multi-seed analysis script (intervention, leakage) builds its
    per-seed checkpoint filename through this one helper."""
    if seed == 42:
        return f"{checkpoint_name}_best_model.pt"
    return f"{checkpoint_name}_best_model_seed{seed}.pt"


def checkpoint_path(filename):
    """
    Every train_*.py script's torch.save(..., checkpoint_path("xxx.pt")) and
    every analysis script's torch.load(checkpoint_path("xxx.pt")) go through
    this single helper, so the checkpoint directory (CHECKPOINT_DIR) only
    needs to change in one place. Creates CHECKPOINT_DIR if it doesn't exist
    yet, same reasoning as tee_stdout_to_file's log_dir creation above --
    works on a fresh checkout with no saved_models/ directory yet.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


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
