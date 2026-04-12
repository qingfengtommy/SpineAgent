"""Downstream condition prediction with linear probing on SpineFM embeddings.

Extracts embeddings from pretrained DINOv3 encoders (with or without router),
then trains a linear classifier (or MIL-based aggregator) for spinal condition
prediction tasks.

Two embedding modes:
  - Direct encoder: use a single T1 or T2 DINOv3 encoder (--pretrained_weights)
  - Router mode:    use both T1+T2 encoders with the trained synthesizer
                    to produce fused embeddings (--use_router)
"""

import argparse
import os
import pickle as pkl
import sys

import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downstream_tasks.lora import apply_lora_to_model
from downstream_tasks.utils import (
    AttentionPooler,
    AttentionPoolerWithProjectionHead,
    DINOv3Dataset,
    DINOv3DatasetCache,
    LinearClassifier,
    LinearProjection,
    MeanLinearClassifier,
    MILLinearClassifier,
    PoolerWithClassifier,
    evaluate,
    patient_collate_fn,
    patient_collate_fn_pad,
    seed_all,
    split_data,
)

torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------
# Model building helpers
# ---------------------------------------------------------------------------

def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    return torch.float


def _build_single_encoder(config_file, pretrained_weights):
    """Load one DINOv3 encoder from config + checkpoint."""
    from dinov3.configs.config import DinoV3SetupArgs, get_cfg_from_args
    from dinov3.models import build_model_from_cfg
    from dinov3.checkpointer import init_model_from_checkpoint_for_evals

    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=pretrained_weights,
        output_dir="",
    )
    cfg = get_cfg_from_args(setup_args)
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    autocast_dtype = get_autocast_dtype(cfg)
    model.to_empty(device="cuda")
    init_model_from_checkpoint_for_evals(model, pretrained_weights, "teacher")
    model.eval()
    return model, autocast_dtype


def build_model_for_eval(config_file, pretrained_weights, lora_weights=None):
    """Build a single DINOv3 encoder for direct (non-router) evaluation."""
    model, autocast_dtype = _build_single_encoder(config_file, pretrained_weights)

    if lora_weights is not None:
        lora_layers = apply_lora_to_model(model)
        lora_ckpt = torch.load(lora_weights, map_location="cpu")
        for name, lora_layer in lora_layers.items():
            lora_layer.load_state_dict(lora_ckpt["t1_lora_state_dict"][name], strict=True)
        model.eval()

    return model, autocast_dtype


def build_router_for_eval(args):
    """Build the SynthesizerEncoder that fuses T1+T2 via the trained router."""
    from model_alignment.models import LayerWiseSynthesizer, SynthesizerEncoder

    t1_encoder, autocast_dtype = _build_single_encoder(args.t1_config, args.t1_checkpoint)
    t2_encoder, _ = _build_single_encoder(args.t2_config, args.t2_checkpoint)

    num_layers = len(t1_encoder.blocks)
    embed_dim = t1_encoder.embed_dim
    synthesizer = LayerWiseSynthesizer(num_layers=num_layers, embed_dim=embed_dim)

    router_ckpt = torch.load(args.router_checkpoint, map_location="cpu")
    synthesizer.load_state_dict(router_ckpt["synthesizer_state"])

    encoder = SynthesizerEncoder(t1_encoder, t2_encoder, synthesizer).cuda()
    encoder.eval()
    return encoder, autocast_dtype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_valid_keys(ckpt_keys, target_keys):
    valid_keys = []
    for key in ckpt_keys:
        if all(t in key for t in target_keys):
            valid_keys.append(key)
    assert len(valid_keys) == 1, f"Expected 1 valid key, got {len(valid_keys)}"
    return valid_keys[0]


def get_embeddings(args, dataset, device, model, autocast_dtype):
    model.eval()
    print("Extracting embeddings...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ret_embeddings = {}
    for iids, images, _ in tqdm(loader, desc="Embedding extraction"):
        images = images.to(device)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                features = model(images)
            features = features.detach().cpu().numpy()
            for i in range(len(iids)):
                ret_embeddings[iids[i]] = features[i]

    return ret_embeddings


def main(args):
    CANDIDATE_TASKS = [
        "par", "canal", "rsna2024", "compression", "edema",
        "inflammation", "neuroforaminal", "spondylolisthesis", "cyst",
        "herniation", "hemangioma", "lipomatosis", "lesion", "fracture",
        "disc_height", "subarticular", "synovial",
    ]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]) if args.normalize else transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    seed_all(args.seed)
    autocast_dtype = None

    if os.path.exists(args.train_split_path) and os.path.exists(args.test_split_path):
        train_slice_path = args.train_split_path.replace(".json", "_slice.json")
        test_patient_path = args.test_split_path.replace(".json", "_patient.json")
        if not (os.path.exists(train_slice_path) and os.path.exists(test_patient_path)):
            train_patient_path, train_slice_path, test_patient_path, test_slice_path = split_data(args)
        else:
            test_slice_path = args.test_split_path.replace(".json", "_slice.json")
            train_patient_path = args.train_split_path.replace(".json", "_patient.json")
    else:
        train_patient_path, train_slice_path, test_patient_path, test_slice_path = split_data(args)

    cache_home = args.cache_home
    task_name = None
    task_num = 0
    for candidate in CANDIDATE_TASKS:
        if candidate in args.json_path.lower():
            task_name = candidate
            task_num += 1
    assert task_num == 1, f"Exactly one task expected, got {task_num} for {args.json_path}"

    if "t1_t2" in args.json_path.lower():
        task_name += "_t1_t2"
    elif "t2" in args.json_path.lower():
        task_name += "_t2"

    model_name = args.log_dir.split("/")[-1]
    cache_dir = os.path.join(cache_home, task_name, model_name)
    os.makedirs(cache_dir, exist_ok=True)

    if task_name == "rsna2024":
        cache_train_file = "rsna2024_train.pkl"
        cache_test_file = "rsna2024_train.pkl"
    else:
        cache_train_file = args.train_split_path.split("/")[-1].replace(".json", ".pkl")
        cache_test_file = args.test_split_path.split("/")[-1].replace(".json", ".pkl")
    cache_train_path = os.path.join(cache_dir, cache_train_file)
    cache_test_path = os.path.join(cache_dir, cache_test_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.flush and os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
        print(f"Loading cached embeddings from {cache_dir}")
        with open(cache_train_path, "rb") as f:
            train_embeddings = pkl.load(f)
        with open(cache_test_path, "rb") as f:
            test_embeddings = pkl.load(f)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pretraining"))
        if args.use_router:
            print("Building router (synthesizer) encoder for fused embeddings...")
            dinov3_model, autocast_dtype = build_router_for_eval(args)
        else:
            dinov3_model, autocast_dtype = build_model_for_eval(
                args.config_file, args.pretrained_weights, args.lora_weights
            )
        for param in dinov3_model.parameters():
            param.requires_grad = False

        if task_name == "rsna2024":
            all_patient_path = args.train_split_path.split("/")
            all_patient_path[-1] = "train_rsna2024_all_split_patient.json"
            all_patient_path = "/".join(all_patient_path)
            all_slice_path = args.train_split_path.split("/")
            all_slice_path[-1] = "train_rsna2024_all_split_slice.json"
            all_slice_path = "/".join(all_slice_path)

            if not os.path.exists(all_patient_path) or not os.path.exists(all_slice_path):
                tmp_json = args.json_path.split("/")
                tmp_json[-1] = "rsna2024_train.json"
                tmp_json = "/".join(tmp_json)
                args.json_path, tmp_json = tmp_json, args.json_path
                all_patient_path, all_slice_path, _, _ = split_data(args, all_in_one=True)
                args.json_path = tmp_json

            train_dataset = DINOv3Dataset(
                base_dir=args.base_dir, transform=preprocess, json_path=all_slice_path, data_format="slice"
            )
            train_embeddings = get_embeddings(args, train_dataset, device, dinov3_model, autocast_dtype)
            with open(cache_train_path, "wb") as f:
                pkl.dump(train_embeddings, f)
        else:
            train_dataset = DINOv3Dataset(
                base_dir=args.base_dir, transform=preprocess, json_path=train_slice_path, data_format="slice"
            )
            test_dataset = DINOv3Dataset(
                base_dir=args.base_dir, transform=preprocess, json_path=test_slice_path, data_format="slice"
            )
            train_embeddings = get_embeddings(args, train_dataset, device, dinov3_model, autocast_dtype)
            test_embeddings = get_embeddings(args, test_dataset, device, dinov3_model, autocast_dtype)
            with open(cache_train_path, "wb") as f:
                pkl.dump(train_embeddings, f)
            with open(cache_test_path, "wb") as f:
                pkl.dump(test_embeddings, f)

        print(f"Embeddings cached to {cache_dir}")
        raise ValueError("Restart the script to load from cache.")

    if args.pool is None:
        train_dataset = DINOv3DatasetCache(
            train_embeddings, base_dir=args.base_dir, json_path=train_slice_path, data_format="slice"
        )
    else:
        args.save_dir = f"{args.save_dir}_{args.pool}"
        args.log_dir = f"{args.log_dir}_{args.pool}"
        args.output_dir = f"{args.output_dir}_{args.pool}"
        train_dataset = DINOv3DatasetCache(
            train_embeddings, base_dir=args.base_dir, json_path=train_patient_path, data_format="patient"
        )
    test_dataset = DINOv3DatasetCache(
        test_embeddings, base_dir=args.base_dir, json_path=test_patient_path, data_format="patient"
    )

    if args.pool is None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=patient_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=patient_collate_fn_pad,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=patient_collate_fn_pad,
        )

    seed_all(args.seed)
    num_classes = args.num_classes
    feature_dim = args.feature_dim
    project_head = None

    if args.project_ckpt is not None:
        if args.projection_type == "slice":
            project_head = LinearProjection(
                in_features=feature_dim, ckpt_path=args.project_ckpt, out_features=1024
            )
            for param in project_head.parameters():
                param.requires_grad = False
            feature_dim = 1024
        ckpt_list = args.project_ckpt.split("/")
        ckpt_name = f"{ckpt_list[-3]}__{ckpt_list[-1].split('.')[0]}"
        args.save_dir = f"{args.save_dir}_{args.projection_type}_{ckpt_name}"
        args.log_dir = f"{args.log_dir}_{args.projection_type}_{ckpt_name}"
        args.output_dir = f"{args.output_dir}_{args.projection_type}_{ckpt_name}"

    if args.pool is None:
        linear_classifier = LinearClassifier(feature_dim, num_classes)
    elif args.pool == "mean":
        linear_classifier = MeanLinearClassifier(feature_dim, num_classes)
    elif args.pool == "MIL":
        linear_classifier = MILLinearClassifier(feature_dim, num_classes)
        if args.project_ckpt is not None:
            ckpt = torch.load(args.project_ckpt, map_location="cpu")
            if args.projection_type == "patient_attn":
                attn_pooler = AttentionPooler(input_dim=feature_dim)
                if "t2" in args.json_path.lower():
                    valid_key = get_valid_keys(ckpt.keys(), ["t2", "mil"])
                    attn_pooler.load_state_dict(ckpt[valid_key], strict=True)
                else:
                    valid_key = get_valid_keys(ckpt.keys(), ["t1", "mil"])
                    attn_pooler.load_state_dict(ckpt[valid_key], strict=True)
                linear_classifier = PoolerWithClassifier(attn_pooler, num_classes)
            elif args.projection_type == "patient_attn_with_projection":
                attn_pooler = AttentionPoolerWithProjectionHead(input_dim=feature_dim)
                if "t2" in args.json_path.lower():
                    mil_key = get_valid_keys(ckpt.keys(), ["t2", "mil"])
                    proj_key = get_valid_keys(ckpt.keys(), ["t2", "proj"])
                    attn_pooler.pooler.load_state_dict(ckpt[mil_key], strict=True)
                    attn_pooler.head.load_state_dict(ckpt[proj_key], strict=True)
                else:
                    mil_key = get_valid_keys(ckpt.keys(), ["t1", "mil"])
                    proj_key = get_valid_keys(ckpt.keys(), ["t1", "proj"])
                    attn_pooler.pooler.load_state_dict(ckpt[mil_key], strict=True)
                    attn_pooler.head.load_state_dict(ckpt[proj_key], strict=True)
                linear_classifier = PoolerWithClassifier(attn_pooler, num_classes)
            elif args.projection_type != "slice":
                raise ValueError(f"Unknown projection type: {args.projection_type}")
            if args.fix_mil:
                for param in linear_classifier.pooler.parameters():
                    param.requires_grad = False
    else:
        raise ValueError(f"Unknown pooling method: {args.pool}")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in linear_classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)

    linear_classifier.to(device)

    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint)
        linear_classifier.load_state_dict(checkpoint["classifier_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=args.log_dir)

    if args.use_router:
        mode = "t1"  # router produces fused features; use T1 projection by default
    elif args.pretrained_weights:
        mode = "t2" if "t2" in args.pretrained_weights else "t1"
    else:
        mode = "t2" if "t2" in args.json_path else "t1"

    if project_head is not None:
        project_head.to(device)

    for epoch in range(start_epoch, args.num_epochs):
        linear_classifier.train()
        running_loss, total_samples, correct_samples = 0.0, 0, 0

        if args.pool is None:
            for i, (features, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.num_epochs}]")
            ):
                features = features.to(device)
                labels = labels.to(device)

                if project_head is not None:
                    with torch.no_grad():
                        features = project_head.forward_t1(features) if mode == "t1" else project_head.forward_t2(features)

                optimizer.zero_grad()
                outputs = linear_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_samples += labels.size(0)
                correct_samples += (torch.argmax(outputs, dim=1) == labels).sum().item()
                writer.add_scalar("Loss/train_step", loss.item(), epoch * len(train_loader) + i)
        else:
            for i, (padded_tensors, masks, labels, _) in enumerate(
                tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.num_epochs}]")
            ):
                padded_tensors = padded_tensors.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                labels[labels < 0] = 0

                if project_head is not None:
                    with torch.no_grad():
                        padded_tensors = (
                            project_head.forward_t1(padded_tensors)
                            if mode == "t1"
                            else project_head.forward_t2(padded_tensors)
                        )

                optimizer.zero_grad()
                outputs = linear_classifier(padded_tensors, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_samples += labels.size(0)
                correct_samples += (torch.argmax(outputs, dim=1) == labels).sum().item()
                writer.add_scalar("Loss/train_step", loss.item(), epoch * len(train_loader) + i)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_samples / total_samples
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_acc, epoch)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        pred_save_path = os.path.join(args.log_dir, f"predictions_epoch_{epoch + 1}.json")
        test_loss, test_acc, test_f1, test_auroc, test_auprc = evaluate(
            mode, linear_classifier, test_loader, criterion, device,
            save_path=pred_save_path, epoch=epoch,
            logits_mode=args.logits_mode, lowest_k=args.lowest_k,
            pool=args.pool, num_classes=args.num_classes,
            project_head=project_head,
        )

        writer.add_scalar("Loss/test_epoch", test_loss, epoch)
        writer.add_scalar("Accuracy/test_epoch", test_acc, epoch)
        writer.add_scalar("F1/test_epoch", test_f1, epoch)
        writer.add_scalar("AUROC/test_epoch", test_auroc if test_auroc else 0.0, epoch)
        writer.add_scalar("AUPRC/test_epoch", test_auprc, epoch)

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "classifier_state_dict": linear_classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            },
            os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
        )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpineFM downstream condition prediction")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json_path", type=str, required=True, help="Path to condition labels JSON")
    parser.add_argument("--train_split_path", type=str, required=True)
    parser.add_argument("--test_split_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for images")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--print_freq", type=int, default=10)

    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--feature_dim", type=int, default=1024, help="DINOv3 ViT-L embed dim")

    parser.add_argument("--save_dir", type=str, default="checkpoints/condition_prediction")
    parser.add_argument("--log_dir", type=str, default="runs/condition_prediction")
    parser.add_argument("--config_file", type=str, default=None, help="DINOv3 config YAML path")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="DINOv3 checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--logits_mode", type=str, default="average", choices=["average", "min_entropy"])
    parser.add_argument("--lowest_k", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--flush", action="store_true", help="Re-extract embeddings")
    parser.add_argument("--cache_home", type=str, default="cache_embeddings")
    parser.add_argument("--pool", type=str, default=None, choices=[None, "mean", "MIL"])
    parser.add_argument("--project_ckpt", type=str, default=None, help="CLIP projection checkpoint")
    parser.add_argument("--projection_type", type=str, default=None,
                        choices=[None, "slice", "patient_attn", "patient_attn_with_projection"])
    parser.add_argument("--fix_mil", action="store_true")
    parser.add_argument("--lora_weights", type=str, default=None)

    # Router / synthesizer arguments
    parser.add_argument("--use_router", action="store_true",
                        help="Use the synthesizer encoder (fused T1+T2) instead of a single encoder")
    parser.add_argument("--t1_config", type=str, default=None, help="DINOv3 config for T1 encoder")
    parser.add_argument("--t1_checkpoint", type=str, default=None, help="Pretrained T1 checkpoint")
    parser.add_argument("--t2_config", type=str, default=None, help="DINOv3 config for T2 encoder")
    parser.add_argument("--t2_checkpoint", type=str, default=None, help="Pretrained T2 checkpoint")
    parser.add_argument("--router_checkpoint", type=str, default=None,
                        help="Trained router/synthesizer checkpoint")

    args = parser.parse_args()
    main(args)
