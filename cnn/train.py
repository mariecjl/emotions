import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from data_load import train_loader, test_loader, class_labels
from model import ArchitectureCNN

# -------------------
# Config
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25
LR = 1e-3
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "architecture_cnn_latest.pth")
SAVE_EVERY = 5  # epochs

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------
# Model / Optim / Loss
# -------------------
model = ArchitectureCNN(num_classes=len(class_labels)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

start_epoch = 0

# -------------------
# Resume if checkpoint exists
# -------------------
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"resumed training from epoch {start_epoch}")

else:
    print("no checkpoint found, starting fresh training")

# -------------------
# Training Loop
# -------------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    correct = total = 0
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / (total / labels.size(0)):.4f}",
            "acc": f"{correct / total:.4f}"
        })

    train_acc = correct / total

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_correct = val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # -------------------
    # Save checkpoint every N epochs
    # -------------------
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_labels": class_labels
        }, CHECKPOINT_PATH)

        print(f"💾 Saved checkpoint at epoch {epoch+1}")

# -------------------
# Final save
# -------------------
torch.save({
    "epoch": EPOCHS - 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "class_labels": class_labels
}, CHECKPOINT_PATH)

print("✅ Training complete & final model saved")
