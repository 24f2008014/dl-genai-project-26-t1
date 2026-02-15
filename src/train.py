import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import src.config as config
import os
from src.models import ResNet, EffNetB0, ScratchCNN
from src.dataset import mashup_collate_fn
from src.utils import seed_worker

def spec_augment(mel, freq_masks=2, freq_mask_width=15, time_masks=2, time_mask_ratio=0.2):
    """
    mel: torch.Tensor shape (B, 1, n_mels, T)
    Returns augmented mel (new tensor).
    freq_mask_width: max width in bins
    time_mask_ratio: max fraction of time frames to mask
    """
    B, C, n_mels, T = mel.shape
    out = mel.clone()
    # freq masks (per sample)
    for i in range(B):
        for _ in range(freq_masks):
            w = np.random.randint(0, freq_mask_width + 1)
            if w == 0: 
                continue
            f0 = np.random.randint(0, max(1, n_mels - w + 1))
            out[i, :, f0:f0+w, :] = 0.0
        # time masks
        max_t_w = max(1, int(T * time_mask_ratio))
        for _ in range(time_masks):
            t_w = np.random.randint(0, max_t_w + 1)
            if t_w == 0:
                continue
            t0 = np.random.randint(0, max(1, T - t_w + 1))
            out[i, :, :, t0:t0+t_w] = 0.0
    return out

def mixup_data(x, y, alpha=0.4, device=None):
    """
    Returns mixed inputs, pairs of targets, and lambda:
      x_mix, y_a, y_b, lam
    x: (B, C, H, W), y: (B,) long
    """
    if device is None:
        device = x.device
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(device)
    x2 = x[index]
    y_a = y
    y_b = y[index]
    x_mix = lam*x + (1 - lam)*x2
    return x_mix, y_a, y_b, lam

def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """
    criterion: your loss function (e.g., nn.CrossEntropyLoss())
    preds: logits (B, C)
    y_a, y_b: long tensors
    lam: float
    """
    return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)

def train_loop(model, loader, loss_fn, optimizer, device, clip_grad_norm=None, use_spec_augment=True, mixup_alpha=0.2):
    model.train()
    running_loss, total = 0, 0
    preds, targets = [], []

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["mel"].to(device)
        y = batch["label"].to(device)
        if use_spec_augment:
            x = spec_augment(x)
        optimizer.zero_grad()

        if mixup_alpha and mixup_alpha > 0:
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
            logits = model(x_mix)
            loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
            # for metrics, use hard labels from y_a (approximate) or skip train-f1/acc while using mixup
            y_pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.append(y_pred)
            targets.append(y.detach().cpu().numpy())
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            y_pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.append(y_pred)
            targets.append(y.detach().cpu().numpy())
        # logits = model(x)
        # loss = loss_fn(logits, y)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        running_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        # y_pred = logits.argmax(dim=1).detach().cpu().numpy()
        # preds.append(y_pred)
        # targets.append(y.detach().cpu().numpy())

    avg_loss = running_loss / total
    preds = np.concat(preds, axis=0)
    targets = np.concat(targets, axis=0)
    train_f1 = f1_score(targets, preds, average="macro")
    train_acc = accuracy_score(targets, preds)
    return avg_loss, train_f1, train_acc

def validate_loop(model, loader, device, loss_fn=None):
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
    
    avg_loss = running_loss / total
    preds = np.concat(preds, axis=0)
    targets = np.concat(targets, axis=0)
    val_f1 = f1_score(targets, preds, average="macro")
    val_f1_class = f1_score(targets, preds, average=None)
    val_acc = accuracy_score(targets, preds)
    return avg_loss, val_f1, val_acc, val_f1_class

def train(
    train_dataset,
    val_dataset,
    model_name="resnet",
    out_dir="output",
    epochs=30,
    batch_size=config.BATCH_SIZE,
    lr=1e-4,
    weight_decay=1e-4,
    device=None,
    num_workers=8,
    seed=42
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mashup_collate_fn,
        worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True,
        collate_fn=mashup_collate_fn,
        worker_init_fn=seed_worker,
    )
    num_classes = len(config.GENRES)

    if model_name == "resnet":
        model = ResNet(num_classes=num_classes, freeze_backbone=False)
    elif model_name == "effnet":
        model = EffNetB0(num_classes=num_classes, pretrained=True, freeze_backbone=False)
    
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1
    os.makedirs(out_dir, exist_ok=True)

    for t in range(1, epochs+1):
        print(f"Epoch: {t}/{epochs}")
        train_loss = train_loop(model, train_dataloader, loss_fn, optimizer, device, clip_grad_norm=2)
        val_f1, class_f1 = validate_loop(model, val_dataloader, device)
        print(f"Train loss: {train_loss:.4f} | Val macro F1: {val_f1:.4f}")
        print(F"Class F1: {class_f1}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model_state": model.state_dict()}, os.path.join(out_dir, "best_ckpt_effnetb0_weighted.ckpt"))
            print("Checkpoint saved.")
    print(f"END. Best val macro F1: {best_val_f1}")
    return model