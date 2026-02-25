import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import src.config as config
import os
from src.models import ResNet, EffNetB0, ScratchCNN, ScratchCRNN, ScratchCNNAttention
from src.dataset import PrecomputedDataset, mashup_collate_fn
from src.utils import seed_worker, get_train_stats
import wandb

def spec_augment(mel: torch.Tensor, freq_masks=2, freq_mask_width=15, time_masks=2, time_mask_ratio=0.2) -> torch.Tensor:
    """Apply SpecAugment to a log-mel spectrogram.
    
    Args:
        mel: A torch.Tensor of shape (B, 1, n_mels, T).
        freq_masks: Number of frequency masks to apply.
        freq_mask_width: Maximum frequency mask width, as a number of bins.
        time_masks: Number of time masks to apply.
        time_mask_ratio: Maximum fraction of time frames to mask.
    
    Returns:
        Augmented mel (new tensor).
    """
    B, C, n_mels, T = mel.shape
    out = mel.clone()

    for i in range(B):
        for _ in range(freq_masks):
            w = np.random.randint(0, freq_mask_width + 1) # Right-open
            if w == 0: 
                continue
            f0 = np.random.randint(0, max(1, n_mels - w + 1)) # Mask location
            out[i, :, f0:f0+w, :] = 0.0
        max_t_w = max(1, int(T * time_mask_ratio))
        for _ in range(time_masks):
            t_w = np.random.randint(0, max_t_w + 1)
            if t_w == 0:
                continue
            t0 = np.random.randint(0, max(1, T - t_w + 1))
            out[i, :, :, t0:t0+t_w] = 0.0
    return out

def mixup(x: torch.Tensor, y, alpha=0.4, device=None):
    """Returns mixed inputs, pairs of targets, and lambda.

    Args:
        x: Shape (B, C, H, W).
        y: Shape (B,).

    Returns:
        x_mix, y_a, y_b, lam
    """
    if device is None:
        device = x.device
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(device)
    x2 = x[index]
    y_a = y
    y_b = y[index]
    x_mix = (lam * x) + ((1 - lam) * x2)
    return x_mix, y_a, y_b, lam

def train_loop(
        model: torch.nn.Module,
        loader: DataLoader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        device,
        clip_grad_norm=None,
        use_spec_augment=True,
        mixup_alpha=0.2
    ):
    """Train a pytorch model for one epoch.

    Returns:
        train_loss, train_f1, train_acc    
    """
    model.train()
    running_loss, total = 0, 0
    preds, targets = [], []

    for batch in tqdm(loader, desc="train", leave=False): # Progress bar disappears when done
        x = batch["mel"].to(device)
        y = batch["label"].to(device)

        if use_spec_augment:
            x = spec_augment(x)
        
        optimizer.zero_grad()
        if mixup_alpha and mixup_alpha > 0:
            x_mix, y_a, y_b, lam = mixup(x, y, mixup_alpha, device)
            logits = model(x_mix)
            loss = (lam * loss_fn(logits, y_a)) + ((1 - lam) * loss_fn(logits, y_b))
            y_pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.append(y_pred)
            targets.append(y.detach().cpu().numpy())
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            y_pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.append(y_pred)
            targets.append(y.detach().cpu().numpy())
        
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        running_loss += float(loss.item()) * x.size(0)
        total += x.size(0)

    train_loss = running_loss / total
    preds = np.concat(preds, axis=0)
    targets = np.concat(targets, axis=0)
    train_f1 = f1_score(targets, preds, average="macro")
    train_acc = accuracy_score(targets, preds)
    return train_loss, train_f1, train_acc

def validate_loop(model: torch.nn.Module, loader: DataLoader, device, loss_fn=None):
    """Validate a pytorch model.

    Returns:
        val_loss, val_f1, val_acc, val_f1_class    
    """
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    running_loss, total = 0, 0
    preds, targets = [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="val", leave=False):
            x = batch["mel"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            running_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
            y_pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.append(y_pred)
            targets.append(y.detach().cpu().numpy())
    
    val_loss = running_loss / total
    preds = np.concat(preds, axis=0)
    targets = np.concat(targets, axis=0)
    val_f1 = f1_score(targets, preds, average="macro")
    val_f1_class = f1_score(targets, preds, average=None)
    val_acc = accuracy_score(targets, preds)
    return val_loss, val_f1, val_acc, val_f1_class

def make_optimizer(
    model: torch.nn.Module,
    backbone_lr=1e-4,
    head_lr=3e-4,
    weight_decay=1e-4,
    lr=3e-4,
    single_lr=False,
    ):
    """Return a new AdamW optimizer."""
    if single_lr:
        return torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def train_sweep_run(model_name="effnet", n_epochs=10, num_workers=8, batch_size=16):
    """Log one run as part of a wandb HPT sweep."""
    run = wandb.init()
    cfg = wandb.config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mean, std = get_train_stats(config.TRAIN_STATS_PATH)
    train_set = PrecomputedDataset("precomputed/train_v4_list.txt", mean=mean, std=std)
    val_set = PrecomputedDataset("precomputed/val_v4_list.txt", mean=mean, std=std)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mashup_collate_fn,
        worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True,
        collate_fn=mashup_collate_fn,
        worker_init_fn=seed_worker,
    )

    if model_name == "effnet":
        model = EffNetB0(num_classes=10, pretrained=True, freeze_backbone=cfg.freeze_backbone)
    elif model_name == "resnet":
        model = ResNet(num_classes=10, pretrained=True, freeze_backbone=cfg.freeze_backbone)
    elif model_name == "scratch":
        model = ScratchCNN(num_classes=10, in_ch=1, base_ch=32)
    elif model_name == "crnn":
        model = ScratchCRNN(num_classes=10, lstm_hidden=128, lstm_layers=1)
    elif model_name == "scratch_attention":
        model = ScratchCNNAttention(num_classes=10)
    model = model.to(device)

    # Single LR for ScratchCNN; different LRs for backbone and head for resnet/effnet
    if (model_name == "scratch") or (model_name == "crnn") or (model_name == "scratch_attention"):
        optimizer = make_optimizer(model=model, lr=cfg.lr, single_lr=True)
    else:
        optimizer = make_optimizer(model=model, backbone_lr=cfg.backbone_lr, head_lr=cfg.head_lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(config.OUT_DIR, exist_ok=True)
    best_val_f1 = -1

    for t in range(1, n_epochs+1):
        print(f"Epoch: {t}/{n_epochs}")
        train_loss, train_f1, train_acc = train_loop(model, train_dataloader, loss_fn, optimizer, device, clip_grad_norm=2)
        val_loss, val_f1, val_acc, val_f1_class = validate_loop(model, val_dataloader, device)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val macro F1: {val_f1:.4f}")
        print(F"Class F1: {val_f1_class}")

        scheduler.step(val_loss)

        # If unfreezing, easier to just build a new optimizer and scheduler than fiddle with param_groups.
        if cfg.freeze_backbone and t == cfg.unfreeze_epoch:
            for p in model.backbone.parameters(): # But keep BatchNorm2d layers still on eval mode
                p.requires_grad = True
                optimizer = make_optimizer(model, cfg.backbone_lr, cfg.head_lr, cfg.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        wandb.log({
            "epoch": t,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/macro_f1": train_f1,
            "val/macro_f1": val_f1,
            "train/acc": train_acc,
            "val/acc": val_acc
        })
        best_val_f1 = max(val_f1, best_val_f1)

    print(f"END. Best val macro F1: {best_val_f1:.4f}")
    run.finish()

def train_run(model_name, i, cfg, n_epochs=30, num_workers=8, batch_size=16):
    """Log a single standalone run with wandb."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mean, std = get_train_stats(config.TRAIN_STATS_PATH)
    train_set = PrecomputedDataset("precomputed/train_v4_list.txt", mean=mean, std=std)
    val_set = PrecomputedDataset("precomputed/val_v4_list.txt", mean=mean, std=std)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mashup_collate_fn,
        worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True,
        collate_fn=mashup_collate_fn,
        worker_init_fn=seed_worker,
    )

    if model_name == "effnet":
        model = EffNetB0(num_classes=10, pretrained=True, freeze_backbone=cfg["freeze_backbone"])
    elif model_name == "resnet":
        model = ResNet(num_classes=10, pretrained=True, freeze_backbone=cfg["freeze_backbone"])
    elif model_name == "scratch":
        model = ScratchCNN(num_classes=10, in_ch=1, base_ch=32)
    elif model_name == "crnn":
        model = ScratchCRNN(num_classes=10, lstm_hidden=128, lstm_layers=1)
    elif model_name == "scratch_attention":
        model = ScratchCNNAttention(num_classes=10)
    model = model.to(device)

    # Single LR for ScratchCNN; different LRs for backbone and head for resnet/effnet
    if (model_name == "scratch") or (model_name == "crnn") or (model_name == "scratch_attention"):
        optimizer = make_optimizer(model=model, lr=cfg["lr"], head_lr=cfg["lr"], backbone_lr=cfg["lr"], weight_decay=cfg["weight_decay"], single_lr=True)
    else:
        optimizer = make_optimizer(model=model, backbone_lr=cfg["backbone_lr"], head_lr=cfg["head_lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(config.OUT_DIR, exist_ok=True)
    best_val_f1 = -1

    for t in range(1, n_epochs+1):
        print(f"Epoch: {t}/{n_epochs}")
        train_loss, train_f1, train_acc = train_loop(model, train_dataloader, loss_fn, optimizer, device, clip_grad_norm=2)
        val_loss, val_f1, val_acc, val_f1_class = validate_loop(model, val_dataloader, device)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val macro F1: {val_f1:.4f}")
        print(F"Class F1: {val_f1_class}")

        scheduler.step(val_loss)

        # If unfreezing, easier to just build a new optimizer and scheduler than fiddle with param_groups.
        if cfg["freeze_backbone"] and t == cfg["unfreeze_epoch"]:
            for p in model.backbone.parameters(): # But keep BatchNorm2d layers still on eval mode
                p.requires_grad = True
                optimizer = make_optimizer(model=model, backbone_lr=cfg["backbone_lr"], head_lr=cfg["head_lr"], weight_decay=cfg["weight_decay"])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        wandb.log({
            "epoch": t,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/macro_f1": train_f1,
            "val/macro_f1": val_f1,
            "train/acc": train_acc,
            "val/acc": val_acc
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model_state": model.state_dict()}, os.path.join(config.OUT_DIR, f"{model_name}_{i}.ckpt"))
            print("Checkpoint saved.")

    print(f"END. Run {i} best val macro F1: {best_val_f1:.4f}")