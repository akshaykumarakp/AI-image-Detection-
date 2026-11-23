"""
Data loader tailored for CIFAKE and custom image datasets.
CIFAKE structure from Kaggle:
  cifake/ (or dataset/)
    train/
      REAL/ (or real/)
      FAKE/ (or fake/)
    test/ (or val/)
      REAL/ (or real/)
      FAKE/ (or fake/)
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size=32):
    """
    CIFAKE images are natively 32x32.
    Using img_size=32 or 64 gives super-fast training and high accuracy.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    return train_transforms, val_transforms

def find_split_dir(data_dir, split_candidates):
    for candidate in split_candidates:
        path = os.path.join(data_dir, candidate)
        if os.path.exists(path):
            return path
    return None

def get_dataloaders(data_dir="dataset", batch_size=64, img_size=32, num_workers=2):
    # Support both train/val and train/test naming conventions
    train_dir = find_split_dir(data_dir, ["train", "Train", "TRAIN"])
    val_dir = find_split_dir(data_dir, ["test", "Test", "TEST", "val", "Val", "validation"])

    if not train_dir:
        raise FileNotFoundError(f"Could not find training directory in '{data_dir}' (looked for 'train')")
    if not val_dir:
        raise FileNotFoundError(f"Could not find validation/test directory in '{data_dir}' (looked for 'test' or 'val')")

    train_tf, val_tf = get_transforms(img_size=img_size)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    # Class names usually: ['FAKE', 'REAL']
    return train_loader, val_loader, train_dataset.classes
