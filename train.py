"""
Training script optimized for Kaggle CIFAKE dataset.
Supports batch sizes, GPU/CPU auto-detection, learning rate scheduling.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from model import build_model

def train(data_dir="dataset", epochs=10, batch_size=64, lr=0.001, model_type="resnet18", save_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f" Training on Device: {device}")
    print(f" Dataset Path:       {data_dir}")
    print(f" Model Architecture: {model_type}")
    print(f" Epochs:             {epochs}")
    print(f" Batch Size:         {batch_size}")
    print(f"==========================================\n")

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        img_size=32,
        num_workers=0  # Safe default on Windows to prevent multi-threading lock
    )
    print(f"Detected Classes: {class_names}")

    # Build model
    model = build_model(model_type=model_type, num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        # Validation Phase
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = val_correct / total_val if total_val > 0 else 0

        print(f"Epoch {epoch}/{epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

        # Save best weights
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint = {
                "model_type": model_type,
                "model_state": model.state_dict(),
                "classes": class_names,
                "best_acc": best_val_acc,
                "img_size": 32
            }
            torch.save(checkpoint, save_path)
            print(f"  ★ New best model saved to {save_path} ({best_val_acc*100:.2f}% Val Acc)")

    print(f"\nTraining Finished! Top Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detector on CIFAKE Dataset")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to CIFAKE root folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (5-10 recommended)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (64 or 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "simple"], help="Model type")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Checkpoint save destination")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model,
        save_path=args.save_path
    )
