"""Transfer-learning image classifier (ResNet) for medical images + evaluation + Grad-CAM samples.
Works with any ImageFolder-style dataset:   <data>/Training/<class>/*.jpg   and   <data>/Testing/<class>/*.jpg
(also train/ + test/).  Examples: Brain Tumor MRI (4 classes), Chest X-Ray Pneumonia (2 classes).
Usage:  python train.py --data /path/to/dataset --epochs 8 --arch resnet50"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.model import build_model, eval_transform, train_transform
from src.gradcam import GradCAM
from src.utils import find_split_dirs, overlay_heatmap

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--head_epochs", type=int, default=2, help="epochs with frozen backbone")
    p.add_argument("--epochs", type=int, default=8, help="epochs of full fine-tuning")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=2, help="use 0 on Windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    p.add_argument("--model_out", default="models/best_model.pt")
    return p.parse_args()

def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0); correct += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return loss_sum / n, correct / n

def predict_all(model, loader, device):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist(); labels += y.tolist()
    return np.array(labels), np.array(preds)

def set_backbone_trainable(model, flag):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = flag

def main():
    a = parse()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.dirname(a.model_out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(CPU: training will be slow, use Kaggle/Colab GPU)")

    train_dir, test_dir = find_split_dirs(a.data)
    full_train = datasets.ImageFolder(train_dir, train_transform())
    full_eval = datasets.ImageFolder(train_dir, eval_transform())      # same files, no augmentation (for validation)
    test_ds = datasets.ImageFolder(test_dir, eval_transform())
    classes = full_train.classes
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(0.15 * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_ds, val_ds = Subset(full_train, tr_idx), Subset(full_eval, val_idx)
    print(f"classes: {classes} | train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    mk = lambda ds, shuf: DataLoader(ds, batch_size=a.batch, shuffle=shuf, num_workers=a.workers, pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    # class weights (inverse frequency) to handle imbalance
    counts = np.bincount([full_train.targets[i] for i in tr_idx], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(a.arch, len(classes)).to(device)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state = 0.0, None

    def fit(epochs, optimizer, scheduler=None, tag=""):
        nonlocal best_val, best_state
        for ep in range(epochs):
            t0 = time.time()
            tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
            vl, va = run_epoch(model, val_dl, criterion, device)
            if scheduler: scheduler.step()
            for k, v in zip(hist, (tl, vl, ta, va)): hist[k].append(v)
            if va > best_val:
                best_val = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{tag} {ep+1}/{epochs}] train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s")

    # Phase 1: train only the new classifier head
    set_backbone_trainable(model, False)
    fit(a.head_epochs, torch.optim.AdamW(model.fc.parameters(), lr=1e-3), tag="head")
    # Phase 2: fine-tune the whole network at a lower learning rate
    set_backbone_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    fit(a.epochs, opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs), tag="finetune")

    # Save best (by validation accuracy) and evaluate on the untouched test set
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes, "arch": a.arch}, a.model_out)
    y, p = predict_all(model, test_dl, device)
    report = classification_report(y, p, target_names=classes, output_dict=True, digits=4)
    print("\n=== TEST SET RESULTS ===\n" + classification_report(y, p, target_names=classes, digits=4))
    summary = {"arch": a.arch, "classes": classes, "best_val_acc": round(best_val, 4),
               "test_accuracy": round(float((y == p).mean()), 4),
               "test_macro_f1": round(float(f1_score(y, p, average="macro")), 4),
               "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")} for c in classes}}
    json.dump(summary, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    # Plots: confusion matrix + learning curves
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right"); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].plot(hist["train_loss"], label="train"); axs[0].plot(hist["val_loss"], label="val"); axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(hist["train_acc"], label="train"); axs[1].plot(hist["val_acc"], label="val"); axs[1].set_title("Accuracy"); axs[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "training_curves.png"), dpi=130); plt.close(fig)

    # Grad-CAM examples on random test images
    cam_gen, tf = GradCAM(model, model.layer4[-1]), eval_transform()
    pick = random.Random(a.seed).sample(range(len(test_ds)), min(6, len(test_ds)))
    fig, axs = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.4))
    for col, i in enumerate(pick):
        path, true = test_ds.samples[i]
        img = Image.open(path).convert("RGB")
        cam, pred, probs = cam_gen(tf(img).unsqueeze(0).to(device))
        axs[0, col].imshow(img); axs[0, col].set_title(f"true: {classes[true]}", fontsize=8)
        axs[1, col].imshow(overlay_heatmap(img, cam)); axs[1, col].set_title(f"pred: {classes[pred]} ({probs[pred]:.0%})", fontsize=8,
                                                                            color="green" if pred == true else "red")
        axs[0, col].axis("off"); axs[1, col].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "gradcam_examples.png"), dpi=130); plt.close(fig)
    print(f"\nSaved model -> {a.model_out}\nSaved metrics + plots -> {a.out}/")

if __name__ == "__main__":
    main()
