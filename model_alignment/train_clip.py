"""CLIP-based text-image alignment for SpineFM.

Aligns T1 and T2 DINOv3 encoder representations with clinical report
embeddings from BiomedBERT using a symmetric contrastive objective.
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_alignment.dataset import (
    SpineMRIEmbeddingDataset,
    alignment_collate_fn,
)
from model_alignment.models import (
    AttentionPooler,
    CLIPModel,
    LinearProjectionHead,
    TextProjectionHead,
    clip_contrastive_loss,
)

logger = logging.getLogger(__name__)


def build_text_encoder(model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def extract_text_features(
    tokenizer, text_encoder, texts, device, max_length=512
):
    """Extract [CLS] token features from BiomedBERT."""
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state[:, 0, :]


def train_one_epoch(
    clip_model,
    text_encoder,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    writer,
):
    clip_model.train()
    text_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        t1_embs = batch["t1_embeddings"].to(device)
        t1_mask = batch["t1_mask"].to(device)
        t2_embs = batch["t2_embeddings"].to(device)
        t2_mask = batch["t2_mask"].to(device)
        reports = batch["reports"]

        text_features = extract_text_features(
            tokenizer, text_encoder, reports, device
        )

        logits_per_image, logits_per_text = clip_model(
            t1_embs, t1_mask, t2_embs, t2_mask, text_features
        )
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Loss/train_step", loss.item(), global_step)

    avg_loss = total_loss / max(num_batches, 1)
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    return avg_loss


def main(args):
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    logger.info("Building text encoder...")
    tokenizer, text_encoder = build_text_encoder(args.text_encoder_name)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_embed_dim = text_encoder.config.hidden_size

    logger.info("Building CLIP alignment model...")
    # AttentionPooler outputs pooler_hidden_dim; LinearProjectionHead must match.
    vision_proj = LinearProjectionHead(
        in_features=args.pooler_hidden_dim, out_features=args.proj_dim
    )
    text_proj = TextProjectionHead(
        in_features=text_embed_dim, out_features=args.proj_dim
    )
    t1_pooler = AttentionPooler(input_dim=args.vision_embed_dim, hidden_dim=args.pooler_hidden_dim)
    t2_pooler = AttentionPooler(input_dim=args.vision_embed_dim, hidden_dim=args.pooler_hidden_dim)

    clip_model = CLIPModel(
        vision_proj=vision_proj,
        text_proj=text_proj,
        t1_pooler=t1_pooler,
        t2_pooler=t2_pooler,
        temperature=args.temperature,
    ).to(device)

    if args.distributed:
        clip_model = DDP(clip_model, device_ids=[local_rank])

    logger.info("Loading datasets...")
    dataset = SpineMRIEmbeddingDataset(
        data_json=args.data_json,
        t1_embedding_dir=args.t1_embedding_dir,
        t2_embedding_dir=args.t2_embedding_dir,
        report_csv=args.report_csv,
        max_slices=args.max_slices,
    )

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=alignment_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        clip_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(dataloader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            clip_model,
            text_encoder,
            tokenizer,
            dataloader,
            optimizer,
            scheduler,
            device,
            epoch,
            writer,
        )
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if local_rank == 0:
            model_to_save = clip_model.module if args.distributed else clip_model
            checkpoint = {
                "epoch": epoch,
                "clip_model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "proj_t1_state": model_to_save.vision_proj.proj_t1.state_dict(),
                "proj_t2_state": model_to_save.vision_proj.proj_t2.state_dict(),
                "t1_mil_state": model_to_save.t1_pooler.state_dict(),
                "t2_mil_state": model_to_save.t2_pooler.state_dict(),
                "text_proj_state": model_to_save.text_proj.state_dict(),
            }
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

            latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
            torch.save(checkpoint, latest_path)

    writer.close()
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="CLIP alignment for SpineFM")
    parser.add_argument("--data_json", type=str, required=True,
                        help="Path to patient data JSON")
    parser.add_argument("--report_csv", type=str, required=True,
                        help="Path to clinical reports CSV")
    parser.add_argument("--t1_embedding_dir", type=str, required=True,
                        help="Directory with T1 encoder embeddings")
    parser.add_argument("--t2_embedding_dir", type=str, required=True,
                        help="Directory with T2 encoder embeddings")
    parser.add_argument("--output_dir", type=str, default="checkpoints/clip_alignment")
    parser.add_argument("--text_encoder_name", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    parser.add_argument("--vision_embed_dim", type=int, default=1024,
                        help="DINOv3 ViT-L embedding dimension")
    parser.add_argument("--pooler_hidden_dim", type=int, default=256,
                        help="AttentionPooler hidden/output dim; must match LinearProjectionHead in_features")
    parser.add_argument("--proj_dim", type=int, default=1024,
                        help="Shared projection space dimension")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_slices", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")

    args = parser.parse_args()
    main(args)
