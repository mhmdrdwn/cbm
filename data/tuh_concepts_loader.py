import torch
from torch.utils.data import Dataset

from data.concept_cache import get_raw_concepts


class TUHWithConceptsDataset(Dataset):
    """
    Wraps TUHEndToEndDataset, adding each subject's cached RAW clinical
    EEG concepts (models/concept_bottleneck.py, data/concept_cache.py)
    alongside the existing raw_eeg/label/etc fields.
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
        sid = item["subject_id"]
        x = item["raw_eeg"].squeeze(0)  # (n_ch, T)
        concepts_raw = get_raw_concepts(sid, x, self.sfreq, self.concept_cache_dir)
        return {**item, "concepts_raw": concepts_raw}


def collate_tuh_concepts(items):
    """Same padding/length convention as data/tuh_e2e_loader.py's collate_tuh_e2e,
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
    return {"raw_eeg": batch_raw, "label": labels, "lengths": lengths, "concepts_raw": concepts_raw}
