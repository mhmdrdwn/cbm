import os

import torch

from models.concept_bottleneck import compute_concepts_raw


def concept_cache_path(cache_dir, sid):
    return os.path.join(cache_dir, f"{sid}_concepts.pt")


def get_raw_concepts(sid, x, sfreq, cache_dir):
    """
    x: (n_channels, n_samples) raw signal (numpy or torch). Returns a
    (28,) float32 torch tensor of RAW (family-1-unnormalized) concepts --
    see compute_concepts_raw's docstring for why normalization is deferred.
    Compute-once-then-cache: subsequent calls for the same sid just load
    from disk instead of recomputing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = concept_cache_path(cache_dir, sid)
    if os.path.exists(path):
        return torch.load(path)["concepts_raw"]

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    c = compute_concepts_raw(x, sfreq=sfreq)
    c_t = torch.tensor(c, dtype=torch.float32)
    torch.save({"concepts_raw": c_t}, path)
    return c_t
