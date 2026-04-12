import json
import os
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class SpineMRIAlignmentDataset(Dataset):
    """Dataset for CLIP-style text-image alignment training.

    Each sample represents a patient study with:
    - T1-weighted MRI slices
    - T2-weighted MRI slices
    - Associated clinical report text
    """

    def __init__(
        self,
        data_json: str,
        report_csv: str,
        max_slices: int = 64,
        transform: Optional[transforms.Compose] = None,
    ):
        with open(data_json) as f:
            self.data = json.load(f)

        self.patient_ids = list(self.data.keys())
        self.reports = self._load_reports(report_csv)
        self.max_slices = max_slices

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:
            self.transform = transform

    def _load_reports(self, report_csv: str) -> dict:
        import csv
        reports = {}
        with open(report_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patient_id", row.get("PatientID", ""))
                text = row.get("report", row.get("Report", ""))
                reports[pid] = text
        return reports

    @staticmethod
    def _pad_and_resize(img: Image.Image, target_size: int = 256) -> Image.Image:
        width, height = img.size
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2
        pad_right = max_dim - width - pad_left
        pad_bottom = max_dim - height - pad_top
        img_padded = ImageOps.expand(
            img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
        )
        return img_padded.resize((target_size, target_size), Image.BICUBIC)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = self._pad_and_resize(img)
        return self.transform(img)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        entry = self.data[pid]

        t1_paths = entry.get("t1", [])
        t2_paths = entry.get("t2", [])

        if len(t1_paths) > self.max_slices:
            t1_paths = random.sample(t1_paths, self.max_slices)
        if len(t2_paths) > self.max_slices:
            t2_paths = random.sample(t2_paths, self.max_slices)

        t1_images = [self._load_image(p) for p in t1_paths] if t1_paths else []
        t2_images = [self._load_image(p) for p in t2_paths] if t2_paths else []

        report_text = self.reports.get(pid, "")

        return {
            "patient_id": pid,
            "t1_images": t1_images,
            "t2_images": t2_images,
            "report": report_text,
        }


class SpineMRIEmbeddingDataset(Dataset):
    """Dataset for alignment training using pre-extracted embeddings."""

    def __init__(
        self,
        data_json: str,
        t1_embedding_dir: str,
        t2_embedding_dir: str,
        report_csv: str,
        max_slices: int = 64,
    ):
        with open(data_json) as f:
            self.data = json.load(f)

        self.patient_ids = list(self.data.keys())
        self.t1_embedding_dir = t1_embedding_dir
        self.t2_embedding_dir = t2_embedding_dir
        self.max_slices = max_slices

        import csv
        self.reports = {}
        with open(report_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patient_id", row.get("PatientID", ""))
                text = row.get("report", row.get("Report", ""))
                self.reports[pid] = text

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        entry = self.data[pid]

        t1_embs = self._load_embeddings(entry.get("t1", []), self.t1_embedding_dir)
        t2_embs = self._load_embeddings(entry.get("t2", []), self.t2_embedding_dir)

        report_text = self.reports.get(pid, "")

        return {
            "patient_id": pid,
            "t1_embeddings": t1_embs,
            "t2_embeddings": t2_embs,
            "report": report_text,
        }

    def _load_embeddings(self, paths: list, emb_dir: str) -> torch.Tensor:
        embs = []
        for p in paths[: self.max_slices]:
            key = os.path.splitext(os.path.basename(p))[0]
            emb_path = os.path.join(emb_dir, f"{key}.npy")
            if os.path.exists(emb_path):
                embs.append(torch.from_numpy(np.load(emb_path)))
        if len(embs) == 0:
            return torch.zeros(1, 1024)
        return torch.stack(embs)


def alignment_collate_fn(batch):
    """Collate function that pads variable-length slice embeddings."""
    patient_ids = [s["patient_id"] for s in batch]
    reports = [s["report"] for s in batch]

    t1_list = [s["t1_embeddings"] for s in batch]
    t2_list = [s["t2_embeddings"] for s in batch]

    t1_padded, t1_mask = _pad_embeddings(t1_list)
    t2_padded, t2_mask = _pad_embeddings(t2_list)

    return {
        "patient_ids": patient_ids,
        "t1_embeddings": t1_padded,
        "t1_mask": t1_mask,
        "t2_embeddings": t2_padded,
        "t2_mask": t2_mask,
        "reports": reports,
    }


def _pad_embeddings(emb_list):
    max_len = max(e.shape[0] for e in emb_list)
    dim = emb_list[0].shape[1]
    padded = torch.zeros(len(emb_list), max_len, dim)
    mask = torch.zeros(len(emb_list), max_len, dtype=torch.bool)
    for i, e in enumerate(emb_list):
        n = e.shape[0]
        padded[i, :n] = e
        mask[i, :n] = True
    return padded, mask
