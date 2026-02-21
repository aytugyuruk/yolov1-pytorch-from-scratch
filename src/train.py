import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import YOLOv1
from dataset import Dataset
from loss import YoloLoss

# --- Configuration for training ---
LEARNING_RATE = 1e-4
DEVICE = "mps"
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 10

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_LABEL_DIR = "dataset/train/labels"
VAL_IMG_DIR = "dataset/valid/images"
VAL_LABEL_DIR = "dataset/valid/labels"

train_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


# --- Training Function ---
def train_fn(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    mean_train_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {mean_train_loss}")
    return mean_train_loss

# --- Validation Function ---
def val_fn(val_loader, model, loss_fn):
    model.eval()
    losses = []

    with torch.no_grad():
        loop = tqdm(val_loader, leave=True)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            losses.append(loss.item())
            loop.set_postfix(val_loss=loss.item())

    mean_val_loss = sum(losses) / len(losses)
    print(f"Validation mean loss: {mean_val_loss}")
    return mean_val_loss

def main():
    model = YOLOv1(in_channels=3, split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_dataset = Dataset(transform=train_transform, img_dir=TRAIN_IMG_DIR, label_dir=TRAIN_LABEL_DIR, S=7, B=2, C=1)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2)

    val_dataset = Dataset(transform=val_transform, img_dir=VAL_IMG_DIR, label_dir=VAL_LABEL_DIR, S=7, B=2, C=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2)

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        train_fn(train_loader, model, optimizer, loss_fn)
        val_loss = val_fn(val_loader, model, loss_fn)

        scheduler.step()

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "yolo_v1_best.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

if __name__ == "__main__":
    main()
