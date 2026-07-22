import numpy as np
import torch

# T3/T4/T5/T6 are the legacy 10-20 names for what MNE's standard_1020
# montage (and the modern 10-10 nomenclature) calls T7/T8/P7/P8. NMT (and
# TUH, same channel set) use the legacy names.
_LEGACY_NAME_MAP = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8"}


def get_channel_positions(ch_names):
    """
    3D scalp coordinates for each channel name, from MNE's standard_1020
    montage (case-insensitive, with the legacy T3/T4/T5/T6 -> T7/T8/P7/P8
    mapping) -- verified to cover NMT/TUH's 21 channels (including A1/A2
    mastoid references).

    ch_names: list of str, in the same channel order used by the calling
              dataset/model.

    Returns: (n_ch, 3) float32 tensor, mean-centered and scaled to roughly
    [-1, 1] -- a stable-scale prior for ShallowCNNPhysicsLoss's learnable
    channel_embed (models/shallow_cnn.py).
    """
    import mne
    mne.set_log_level("ERROR")

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_pos = {k.lower(): v for k, v in montage.get_positions()["ch_pos"].items()}

    coords = []
    for name in ch_names:
        key = name.lower()
        key = _LEGACY_NAME_MAP.get(key, key)
        if key not in montage_pos:
            raise KeyError(f"channel '{name}' not found in standard_1020 montage")
        coords.append(montage_pos[key])

    coords = np.stack(coords).astype(np.float32)   # (n_ch, 3), meters, head-centered
    coords = coords - coords.mean(axis=0, keepdims=True)
    coords = coords / (np.abs(coords).max() + 1e-8)  # scale to roughly [-1, 1]
    return torch.tensor(coords, dtype=torch.float32)
