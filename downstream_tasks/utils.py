import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset
from tqdm import tqdm


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DINOv3Dataset(Dataset):
    """Dataset for loading spine MRI images with labels."""

    def __init__(self, base_dir, transform=None, json_path=None, data=None, data_format="slice"):
        if data is not None:
            self.data = data
        elif json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            raise ValueError("Either json_path or data must be provided.")
        self.base_dir = base_dir
        self.transform = transform
        self.data_format = data_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        if self.data_format == "slice":
            image_path = os.path.join(self.base_dir, entry["file"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
            if image.mode == "L":
                image_np = np.array(image)
                image_np = np.stack([image_np] * 3, axis=-1)
                image = Image.fromarray(image_np)
            if self.transform:
                image = self.transform(image)
            label = int(entry["label"])
            return entry["file"], image, label

        elif self.data_format == "patient":
            slices = entry["slices"]
            images = []
            for slice_entry in slices:
                image_path = os.path.join(self.base_dir, slice_entry["file"])
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image = Image.open(image_path)
                if image.mode == "L":
                    image_np = np.array(image)
                    image_np = np.stack([image_np] * 3, axis=-1)
                    image = Image.fromarray(image_np)
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            label = int(slices[0]["label"])
            return {"images": images, "label": label}

        else:
            raise ValueError("data_format must be either 'slice' or 'patient'.")


class DINOv3DatasetCache(Dataset):
    """Dataset that loads pre-extracted embeddings with labels."""

    def __init__(self, embeddings, base_dir, json_path, data_format="slice"):
        if json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            raise ValueError("json_path must be provided.")
        self.embeddings = embeddings
        self.base_dir = base_dir
        self.data_format = data_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        if self.data_format == "slice":
            key = entry["file"]
            feature = self.embeddings[key]
            label = int(entry["label"])
            return torch.from_numpy(feature), label
        elif self.data_format == "patient":
            patient_id = entry["patient_id"]
            slices = entry["slices"]
            features = []
            for slice_entry in slices:
                key = slice_entry["file"]
                feature = self.embeddings[key]
                features.append(feature)
            label = int(slices[0]["label"])
            return {
                "images": torch.from_numpy(np.array(features)),
                "label": label,
                "patient_id": patient_id,
            }
        else:
            raise ValueError("data_format must be either 'slice' or 'patient'.")


class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class LinearProjection(nn.Module):
    """Projection head for transforming embeddings using CLIP-trained weights."""

    def __init__(self, in_features, ckpt_path, out_features=1024):
        super().__init__()
        self.proj_t1 = nn.Linear(in_features, out_features)
        self.proj_t2 = nn.Linear(in_features, out_features)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.proj_t1.load_state_dict(ckpt["proj_t1_state"])
        self.proj_t2.load_state_dict(ckpt["proj_t2_state"])

    def forward_t1(self, x):
        return F.normalize(self.proj_t1(x), dim=-1)

    def forward_t2(self, x):
        return F.normalize(self.proj_t2(x), dim=-1)


class MeanLinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, pad_mask):
        pad_mask_float = pad_mask.float().unsqueeze(-1)
        x = x * pad_mask_float
        x = x.sum(dim=1) / pad_mask_float.sum(dim=1)
        x = self.norm(x)
        return self.classifier(x)


class Attn_Net_Gated(nn.Module):
    """Gated attention network for MIL-based aggregation."""

    def __init__(self, L=1024, D=256, dropout=True, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        return self.attention_c(a * b)


class MILLinearClassifier(nn.Module):
    """Multiple Instance Learning classifier with gated attention pooling."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.local_phi = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.25)
        )
        self.local_attn_pool = Attn_Net_Gated(L=256, D=256, dropout=True, n_classes=1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, pad_mask):
        h_256 = self.local_phi(x)
        A_256 = self.local_attn_pool(h_256).squeeze(dim=2)
        attn_logits = torch.masked_fill(A_256, ~pad_mask, -1e9)
        A_256 = F.softmax(attn_logits, dim=1)
        h = torch.bmm(A_256.unsqueeze(dim=1), h_256).squeeze(dim=1)
        return self.classifier(h)


class AttentionPooler(nn.Module):
    """Attention-based pooler for aggregating variable-length sequences."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout_p: float = 0.25):
        super().__init__()
        self.local_phi = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout_p)
        )
        self.attn_net = Attn_Net_Gated(L=256, D=hidden_dim, dropout=(dropout_p > 0), n_classes=1)

    def forward(self, x: torch.Tensor, pad_mask):
        h_phi = self.local_phi(x)
        B, S, D_phi = h_phi.shape

        if S == 0:
            return torch.zeros(B, D_phi, device=x.device, dtype=x.dtype)

        h_phi_flat = h_phi.reshape(B * S, D_phi)
        attn_logits_flat = self.attn_net(h_phi_flat)
        attn_scores = attn_logits_flat.view(B, S, 1).squeeze(dim=2)

        if pad_mask is not None:
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = torch.masked_fill(attn_scores, ~pad_mask, mask_value)

        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.bmm(attn_weights.unsqueeze(dim=1), h_phi).squeeze(dim=1)


class AttentionPoolerWithProjectionHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.pooler = AttentionPooler(input_dim, hidden_dim=256, dropout_p=0.25)
        self.head = nn.Linear(256, 256)

    def forward(self, x, pad_mask):
        h_aggregated = self.pooler(x, pad_mask)
        return self.head(h_aggregated)


class PoolerWithClassifier(nn.Module):
    def __init__(self, pooler, output_dim: int):
        super().__init__()
        self.pooler = pooler
        self.head = nn.Linear(256, output_dim)

    def forward(self, x, pad_mask):
        h_aggregated = self.pooler(x, pad_mask)
        return self.head(h_aggregated)


def split_data(args, all_in_one=False):
    """Split patient-level data into train/test sets and generate JSON files."""
    with open(args.json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    patient_ids = list(full_data.keys())
    random.seed(42)
    random.shuffle(patient_ids)
    split_idx = int(args.train_ratio * len(patient_ids))
    if all_in_one:
        split_idx = len(patient_ids)
    train_patient_ids = patient_ids[:split_idx]
    test_patient_ids = patient_ids[split_idx:] if not all_in_one else []

    train_patient_data, train_slice_data = [], []
    test_patient_data, test_slice_data = [], []

    for pid in train_patient_ids:
        train_patient_data.append({"patient_id": pid, "slices": full_data[pid]})
        for s in full_data[pid]:
            sc = s.copy()
            sc["patient_id"] = pid
            train_slice_data.append(sc)

    if not all_in_one:
        for pid in test_patient_ids:
            test_patient_data.append({"patient_id": pid, "slices": full_data[pid]})
            for s in full_data[pid]:
                sc = s.copy()
                sc["patient_id"] = pid
                test_slice_data.append(sc)

    train_patient_path = args.train_split_path.replace(".json", "_patient.json")
    test_patient_path = args.test_split_path.replace(".json", "_patient.json")
    train_slice_path = args.train_split_path.replace(".json", "_slice.json")
    test_slice_path = args.test_split_path.replace(".json", "_slice.json")

    os.makedirs(os.path.dirname(args.train_split_path), exist_ok=True)
    with open(train_patient_path, "w", encoding="utf-8") as f:
        json.dump(train_patient_data, f, indent=4)
    with open(train_slice_path, "w", encoding="utf-8") as f:
        json.dump(train_slice_data, f, indent=4)
    if not all_in_one:
        with open(test_patient_path, "w", encoding="utf-8") as f:
            json.dump(test_patient_data, f, indent=4)
        with open(test_slice_path, "w", encoding="utf-8") as f:
            json.dump(test_slice_data, f, indent=4)

    print(f"  Patient-level training file: {train_patient_path}")
    print(f"  Patient-level test file:     {test_patient_path}")
    print(f"  Slice-level training file:   {train_slice_path}")
    print(f"  Slice-level test file:       {test_slice_path}")

    return train_patient_path, train_slice_path, test_patient_path, test_slice_path


def patient_collate_fn(batch):
    return [(s["images"], s["label"]) for s in batch]


def patient_collate_fn_pad(batch):
    patient_ids = [s["patient_id"] for s in batch]
    max_len = max(s["images"].shape[0] for s in batch)
    padded_tensors, masks, labels = [], [], []

    for s in batch:
        images = s["images"]
        padded = torch.zeros(max_len, *images.shape[1:], dtype=images.dtype)
        padded[: images.shape[0]] = images
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[: images.shape[0]] = True
        padded_tensors.append(padded)
        masks.append(mask)
        labels.append(s["label"])

    return (
        torch.stack(padded_tensors),
        torch.stack(masks).bool(),
        torch.tensor(labels, dtype=torch.long),
        patient_ids,
    )


def evaluate(
    mode,
    classifier,
    data_loader,
    criterion,
    device,
    save_path=None,
    epoch=-1,
    logits_mode="average",
    lowest_k=1,
    pool=None,
    num_classes=2,
    project_head=None,
):
    classifier.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_patient_ids, all_probs = [], [], [], []

    with torch.inference_mode():
        if pool is None:
            for batch in tqdm(data_loader, desc="Evaluating", total=len(data_loader)):
                for features, label in batch:
                    logits_list = []
                    for start in range(0, features.shape[0], 128):
                        end = min(start + 128, features.shape[0])
                        inter_features = features[start:end].to(device)
                        logits = classifier(inter_features)
                        logits_list.append(logits.detach())

                    logits = torch.cat(logits_list, dim=0)
                    if logits_mode == "average":
                        aggregated_logits = logits.mean(dim=0)
                    elif logits_mode == "min_entropy":
                        probs = torch.softmax(logits, dim=1)
                        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                        _, indices = torch.topk(-entropies, k=lowest_k, largest=True)
                        aggregated_logits = logits[indices].mean(dim=0) if lowest_k > 1 else logits[indices[0]]
                    else:
                        raise ValueError(f"Invalid logits_mode: {logits_mode}")

                    loss_patient = criterion(
                        aggregated_logits.unsqueeze(0),
                        torch.tensor([label], dtype=torch.long, device=device),
                    )
                    running_loss += loss_patient.item()
                    total += 1

                    pred_label = torch.argmax(aggregated_logits)
                    if pred_label.item() == label:
                        correct += 1

                    all_labels.append(label)
                    all_preds.append(pred_label.item())
                    all_patient_ids.append("")
                    prob = torch.softmax(aggregated_logits, dim=0)
                    all_probs.append(prob[1].item())
        else:
            for _, (padded_tensors, masks, labels, patient_ids) in enumerate(
                tqdm(data_loader, desc="Evaluating", total=len(data_loader))
            ):
                padded_tensors = padded_tensors.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                labels[labels < 0] = 0

                if project_head is not None:
                    with torch.no_grad():
                        if mode == "t1":
                            padded_tensors = project_head.forward_t1(padded_tensors)
                        elif mode == "t2":
                            padded_tensors = project_head.forward_t2(padded_tensors)
                        else:
                            raise NotImplementedError(f"Wrong mode: {mode}")

                aggregated_logits = classifier(padded_tensors, masks)
                loss_patient = criterion(aggregated_logits, labels)
                running_loss += loss_patient.item()
                total += labels.size(0)

                pred_label = torch.argmax(aggregated_logits, dim=1)
                correct += (pred_label == labels).sum().item()
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(pred_label.cpu().numpy().tolist())
                probs = torch.softmax(aggregated_logits, dim=1)
                if num_classes > 2:
                    all_probs.extend(probs.cpu().numpy().tolist())
                else:
                    all_probs.extend(probs[:, 1].cpu().numpy().tolist())
                all_patient_ids.extend(list(patient_ids))

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    f1 = f1_score(all_labels, all_preds, average="weighted")
    try:
        if num_classes > 2:
            labels_binarized = label_binarize(all_labels, classes=list(range(num_classes)))
            all_labels_arr = np.array(labels_binarized)
            all_probs_arr = np.array(all_probs)
        else:
            all_labels_arr = np.array(all_labels)
            all_probs_arr = np.array(all_probs)
        auroc = roc_auc_score(all_labels_arr, all_probs_arr, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"Error computing AUROC: {e}")
        auroc = None
    auprc = average_precision_score(
        all_labels_arr if num_classes > 2 else all_labels, all_probs, average="macro"
    )

    print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
    print(f"F1: {f1:.4f} | AUROC: {(auroc if auroc else 0):.4f} | AUPRC: {auprc:.4f}")

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
        curve_data = {
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc},
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist(), "auc": pr_auc},
        }
    else:
        curve_data = {"roc_curve": {"auc": auroc}, "pr_curve": {"auc": auprc}}

    if save_path is not None:
        predictions = []
        for label, pred, prob, pid in zip(all_labels, all_preds, all_probs, all_patient_ids):
            predictions.append({
                "patient_id": pid,
                "true_label": str(label) if not isinstance(label, np.ndarray) else label.tolist(),
                "pred_label": str(int(pred)),
                "prob_score": str(prob) if not isinstance(prob, np.ndarray) else prob.tolist(),
            })
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"predictions": predictions, "curve_data": curve_data}, f, indent=4)

        if num_classes == 2:
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(os.path.dirname(save_path), f"roc_curve_{epoch}.png"))
            plt.close()

            plt.figure()
            plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(os.path.dirname(save_path), f"pr_curve_{epoch}.png"))
            plt.close()

    return epoch_loss, epoch_acc, f1, auroc, auprc
