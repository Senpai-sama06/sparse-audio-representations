import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from lrac.data.manifests import read_manifest
from lrac.data.preprocessing import load_audio


class AudioDataset(Dataset):
    def __init__(self, manifest, sample_rate, crop_samples=None, limit=None):
        self.rows = read_manifest(manifest)
        if limit:
            # Corpus files are usually speaker-sorted. A prefix would validate on
            # one narrator; use round-robin speaker coverage for bounded probes.
            groups = {}
            for row in self.rows:
                groups.setdefault((row["corpus"], row.get("speaker") or row["id"]), []).append(row)
            ordered = list(groups.values())
            selected, offset = [], 0
            while len(selected) < limit:
                added = False
                for group in ordered:
                    if offset < len(group):
                        selected.append(group[offset])
                        added = True
                        if len(selected) == limit:
                            break
                if not added:
                    break
                offset += 1
            self.rows = selected
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.sample_rate, self.crop_samples = sample_rate, crop_samples

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = torch.from_numpy(load_audio(row, self.sample_rate)).unsqueeze(0)
        if self.crop_samples and x.shape[-1] > self.crop_samples:
            start = int(torch.randint(x.shape[-1] - self.crop_samples + 1, (1,)).item())
            x = x[:, start:start + self.crop_samples]
        return x, row

    def sampler(self, generator=None):
        counts = {}
        keys = [(r["corpus"], r.get("speaker") or r["id"]) for r in self.rows]
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        weights = [1 / counts[key] for key in keys]
        return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def collate(batch):
    lengths = torch.tensor([x.shape[-1] for x, _ in batch])
    target = int(lengths.max())
    audio = torch.stack([torch.nn.functional.pad(x, (0, target - x.shape[-1])) for x, _ in batch])
    return audio, lengths, [row for _, row in batch]
