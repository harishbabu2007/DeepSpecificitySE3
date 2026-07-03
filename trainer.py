import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from pdna_dataset import (
    DeepSpecificityDataset,
    collate_fn,
)

from architecture.model import DeepSpecificitySE3
from torch.amp import autocast, GradScaler

TRAIN_DIR = "./data/processed"
# VAL_DIR = "graphs_valid"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 1
EPOCHS = 200

LR = 3e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
GRAD_CLIP = 5.0

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

wandb.init(
    project="Deep-Specificity",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    },
)


train_dataset = DeepSpecificityDataset(TRAIN_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

model = DeepSpecificitySE3().to(DEVICE)
# for name, module in model.named_children():
#     params = sum(p.numel() for p in module.parameters() if p.requires_grad)
#     print(f"{name:25s} {params:,}")

# for name, p in model.se3.named_parameters():
#     print(f"{name:60s} {p.numel():,}")
# model = torch.compile(model, dynamic=True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

scaler = GradScaler("cuda")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters : {num_params:,}")


best_loss = float("inf")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    pbar = tqdm(loader)

    for batch in pbar:

        optimizer.zero_grad(set_to_none=True)

        coords = batch["coords"].to(device, non_blocking=True)
        node_type = batch["node_type"].to(device, non_blocking=True)
        res_type = batch["res_type"].to(device, non_blocking=True)
        interface = batch["interface"].to(device, non_blocking=True)
        dssr = batch["dssr"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        # adj_mat = batch["adj_mat"].to(device)
        edges = batch["edges"].to(device, non_blocking=True)
        pwm_forward = batch["pwm_forward"].to(device, non_blocking=True)
        pwm_reverse = batch["pwm_reverse"].to(device, non_blocking=True)
        pwm_mask_forward = batch["pwm_mask_forward"].to(device, non_blocking=True)
        pwm_mask_reverse = batch["pwm_mask_reverse"].to(device, non_blocking=True)
        num_dna_nodes = batch["num_dna_nodes"].to(device, non_blocking=True)
        num_bp = batch["num_base_pairs"][0].item()

        # Forward
        with autocast("cuda"):
            logits = model(
                coords=coords,
                node_type=node_type,
                residue_id=res_type,
                interface=interface,
                dssr=dssr,
                mask=mask,
                # adj_mat=adj_mat,
                edges=edges,
                num_dna_nodes=num_dna_nodes,
            )

            assert logits.shape[1] == num_dna_nodes[0].item()
            assert logits.shape[2] == 4

            # Split forward / reverse strand

            pred_forward = logits[:, :num_bp]
            pred_reverse = logits[:, num_bp:]

            pred_reverse = torch.flip(
                pred_reverse,
                dims=[1],
            )

            pred_reverse = pred_reverse[
                ...,
                [3, 2, 1, 0],
            ]

            log_probs_fwd = F.log_softmax(
                pred_forward,
                dim=-1,
            )

            loss_forward = -(pwm_forward * log_probs_fwd).sum(dim=-1)

            loss_forward = (loss_forward * pwm_mask_forward.float()).sum() / (
                pwm_mask_forward.float().sum() + 1e-8
            )

            log_probs_rev = F.log_softmax(
                pred_reverse,
                dim=-1,
            )

            loss_reverse = -(pwm_reverse * log_probs_rev).sum(dim=-1)

            loss_reverse = (loss_reverse * pwm_mask_reverse.float()).sum() / (
                pwm_mask_reverse.float().sum() + 1e-8
            )

            loss = (loss_forward + loss_reverse) / 2.0

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        pbar.set_description(f"Loss {loss.item():.5f}")

    return running_loss / len(loader)

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        DEVICE,
    )

    scheduler.step()

    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )

    print(f"Train Loss : {train_loss:.6f}")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
        },
        os.path.join(
            CHECKPOINT_DIR,
            "latest_model.pt",
        ),
    )

    if train_loss < best_loss:
        best_loss = train_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
            },
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pt",
            ),
        )

        print(f"Best model saved " f"(best Loss = {best_loss:.6f})")

wandb.finish()
