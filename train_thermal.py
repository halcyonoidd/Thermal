"""
Klasifikasi Citra Termal: ICAS vs Non-ICAS
Pipeline: Data -> Model (ResNet18 transfer learning) -> Training -> Evaluasi

Jalankan:  python train_thermal.py
Output (folder outputs/): best_model.pth, metrics.json, training_curves.png,
                          confusion_matrix.png, roc_curve.png, sample_predictions.png
"""
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

# ----------------------------- Konfigurasi -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Coba struktur soal dulu, lalu struktur lokal; pakai yang ditemukan
_CANDIDATES = [
    os.path.join(BASE_DIR, "dataset", "datasets", "thermal_classification_cropped"),
    os.path.join(BASE_DIR, "thermal_classification_cropped"),
]
DATA_PATH = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])
OUT_DIR = os.path.join(BASE_DIR, "outputs")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 7            # early stopping
SEED = 42
CLASS_NAMES = ["Non-ICAS", "ICAS"]   # label 0 = non_icas, 1 = icas

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Agar hasil reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================= PART 1: Data Loading =========================
class ThermalDataset(Dataset):
    """Dataset berisi (path, label); transform diberikan per split."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), label


def load_samples(root):
    """Baca semua gambar: non_icas -> 0, icas -> 1."""
    samples = []
    for label, folder in enumerate(["non_icas", "icas"]):
        folder_path = os.path.join(root, folder)
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(folder_path, name), label))
    return samples


def get_dataloaders():
    samples = load_samples(DATA_PATH)
    labels = [s[1] for s in samples]

    # Split stratified 70/15/15 agar proporsi kelas sama di setiap split
    train_s, temp_s = train_test_split(samples, test_size=0.30,
                                       stratify=labels, random_state=SEED)
    val_s, test_s = train_test_split(temp_s, test_size=0.50,
                                     stratify=[s[1] for s in temp_s],
                                     random_state=SEED)

    # Normalisasi dengan statistik ImageNet (karena memakai model pretrained)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([                       # augmentasi hanya untuk train
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), norm,
    ])
    eval_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])

    train_ds = ThermalDataset(train_s, train_tf)
    val_ds = ThermalDataset(val_s, eval_tf)
    test_ds = ThermalDataset(test_s, eval_tf)

    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    loaders = (DataLoader(train_ds, shuffle=True, **kw),
               DataLoader(val_ds, **kw),
               DataLoader(test_ds, **kw))

    # Penanganan class imbalance: pos_weight = jumlah negatif / jumlah positif
    n_pos = sum(s[1] for s in train_s)
    pos_weight = (len(train_s) - n_pos) / n_pos
    print(f"Train/Val/Test: {len(train_s)}/{len(val_s)}/{len(test_s)} | "
          f"pos_weight={pos_weight:.2f}")
    return (*loaders, pos_weight)


# ======================= PART 2: Model Architecture ====================
def build_model():
    """
    ResNet18 pretrained (transfer learning + fine-tuning seluruh layer).
    Alasan: dataset kecil (950 gambar) -> fitur ImageNet membantu
    generalisasi; ResNet18 cukup ringan sehingga risiko overfitting rendah.
    Head: Dropout + 1 output logit (klasifikasi biner).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 1))
    return model.to(DEVICE)


# ======================= PART 3: Training ==============================
def run_epoch(model, loader, criterion, optimizer=None):
    """Satu epoch; jika optimizer diberikan -> mode training, selain itu eval."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += ((logits > 0).float() == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, pos_weight):
    # BCEWithLogits + pos_weight: loss untuk biner & menangani imbalance
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=DEVICE))
    # AdamW: weight decay (regularisasi) yang benar
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, wait = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()

        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc)):
            hist[k].append(v)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        # Simpan model terbaik + early stopping berdasarkan val loss
        if va_loss < best_val_loss:
            best_val_loss, wait = va_loss, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth"),
                                     map_location=DEVICE))
    return hist


# ======================= PART 4: Evaluasi ==============================
@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, desc="Testing"):
        probs.extend(torch.sigmoid(model(x.to(DEVICE)).squeeze(1)).cpu().numpy())
        labels.extend(y.numpy())
    return np.array(labels), np.array(probs)


def evaluate(model, test_loader):
    y_true, y_prob = predict(model, test_loader)
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": auc(fpr, tpr),
        "confusion_matrix": cm.tolist(),
    }
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_history(hist):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, key, title in zip(ax, ("loss", "acc"), ("Loss", "Accuracy")):
        a.plot(hist[f"train_{key}"], label="Train")
        a.plot(hist[f"val_{key}"], label="Validation")
        a.set_title(title); a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()


@torch.no_grad()
def plot_sample_predictions(model, loader, n=8):
    model.eval()
    x, y = next(iter(loader))
    prob = torch.sigmoid(model(x[:n].to(DEVICE)).squeeze(1)).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 7))
    for i, a in enumerate(axes.flat):
        img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()  # de-normalisasi
        pred = int(prob[i] >= 0.5)
        a.imshow(img); a.axis("off")
        a.set_title(f"True: {CLASS_NAMES[y[i]]}\nPred: {CLASS_NAMES[pred]} ({prob[i]:.2f})",
                    color="green" if pred == y[i] else "red", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sample_predictions.png"), dpi=150)
    plt.close()


# ================================ Main =================================
if __name__ == "__main__":
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders()
    model = build_model()
    history = train(model, train_loader, val_loader, pos_weight)
    plot_history(history)
    evaluate(model, test_loader)
    plot_sample_predictions(model, test_loader)
    print(f"Selesai. Semua hasil tersimpan di folder '{OUT_DIR}/'")