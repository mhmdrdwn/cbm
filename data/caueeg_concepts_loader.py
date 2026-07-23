import torch
from torch.utils.data import Dataset

from data.concept_cache import get_raw_concepts
from models.concept_bottleneck import ASYM_PAIRS_CAUEEG, REGIONS_CAUEEG


class CAUEEGWithConceptsDataset(Dataset):
    """
    Wraps CAUEEGEndToEndDataset, adding each subject's cached RAW clinical EEG
    concepts (models/concept_bottleneck.py, data/concept_cache.py) alongside
    the existing raw_eeg/label/etc fields. Same pattern as TUHWithConceptsDataset
    (data/tuh_concepts_loader.py), except concepts are computed with
    REGIONS_CAUEEG/ASYM_PAIRS_CAUEEG -- CAUEEG's channel order differs from
    both TUH/NMT's and ds004504's (see those constants' docstring in
    models/concept_bottleneck.py).

    Cache key is (subject_id, window_idx), not just subject_id: unlike TUH,
    CAUEEGEndToEndDataset's train split can return several different windows
    for the same subject (see its docstring), each with genuinely different
    raw signal -- caching by subject_id alone would compute one window's
    concepts once and then silently reuse them as the "true" concepts for
    every other window of that subject, which is wrong (those windows are
    different segments of EEG, not the same one repeated).
    """

    def __init__(self, base_dataset, concept_cache_dir, sfreq):
        self.dataset = base_dataset
        self.concept_cache_dir = concept_cache_dir
        self.sfreq = sfreq
        self.subjects = base_dataset.subjects  # passthrough, e.g. for class-weight computation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        cache_key = f"{item['subject_id']}_w{item['window_idx']}"
        x = item["raw_eeg"].squeeze(0)  # (n_ch, T)
        concepts_raw = get_raw_concepts(
            cache_key, x, self.sfreq, self.concept_cache_dir,
            regions=REGIONS_CAUEEG, asym_pairs=ASYM_PAIRS_CAUEEG,
        )
        return {**item, "concepts_raw": concepts_raw}


def collate_caueeg_concepts(items):
    """Same padding/length convention as data/caueeg_e2e_loader.py's collate_caueeg_e2e,
    plus a stacked concepts_raw batch (fixed (28,) shape, no padding needed)."""
    lengths = torch.tensor([item["raw_eeg"].shape[-1] for item in items], dtype=torch.long)
    T_max = int(lengths.max())
    n_ch = items[0]["raw_eeg"].shape[1]
    batch_raw = torch.zeros(len(items), n_ch, T_max, dtype=items[0]["raw_eeg"].dtype)
    for i, item in enumerate(items):
        T_i = item["raw_eeg"].shape[-1]
        batch_raw[i, :, :T_i] = item["raw_eeg"].squeeze(0)
    labels = torch.stack([item["label"] for item in items])
    concepts_raw = torch.stack([item["concepts_raw"] for item in items])
    subject_ids = [item["subject_id"] for item in items]
    return {"raw_eeg": batch_raw, "label": labels, "lengths": lengths,
            "concepts_raw": concepts_raw, "subject_id": subject_ids}
