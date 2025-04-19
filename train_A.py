import os
import wandb
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
from typing import List, Tuple
from torch.cuda.amp import autocast, GradScaler
import argparse

# Part 1: Extraction of iNaturalist data
class INatSplitLoader:
    """
    Loads the iNaturalist data, performs a manual per‐class 80/20 split
    (without external libs), and exposes PyTorch DataLoaders.
    """
    def __init__(
        self,
        train_root: str,
        test_root: str,
        img_size: tuple = (256, 256),
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        
        # Preprocessing on data
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        
        self.batch_size = batch_size
        self.train_root = train_root
        self.num_workers = num_workers
        self.test_root = test_root

    def build(self):
        # 1. Load the complete training folder
        full_ds = datasets.ImageFolder(self.train_root, transform=self.transform)

        # 2. Group indices by class label
        class_to_idxs = {}
        for idx, (_, lbl) in enumerate(full_ds.samples):
            class_to_idxs.setdefault(lbl, []).append(idx)

        # 3. For each class, shuffle and split 80/20
        train_idxs, val_idxs = [], []
        for lbl, idxs in class_to_idxs.items():
            # shuffle in‐place
            random.shuffle(idxs)
            split = int(len(idxs)*0.8)
            train_idxs += idxs[:split]
            val_idxs   += idxs[split:]

        # 4. Load test set untouched
        self.test_ds = datasets.ImageFolder(self.test_root, transform=self.transform)
        
        # 5. Build Subsets
        self.val_ds   = Subset(full_ds, val_idxs)
        self.train_ds = Subset(full_ds, train_idxs)
        
        

    def get_loaders(self):
        """
        Returns (train_loader, val_loader, test_loader).
        Exact same loader settings as before so performance is unaffected.
        """
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

# Part2: CNN class module for INaturalist dataset - includes function for training 

class INatCustomCNN(nn.Module):
    """
    A configurable CNN tailored for classifying iNaturalist images.
    Includes its own training routine with mixed-precision support.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_channels: List[int],
        kernel_sizes: List[int],
        fc_units: int,
        act_fn: nn.Module,
        use_batchnorm: bool,
        drop_p: float,
        opt_type: str,
        learning_rate: float,
        l2_reg: float
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.feature_extractor = self._build_conv_layers(input_shape[0], conv_channels, kernel_sizes, use_batchnorm, act_fn)
        self.flatten_dim = self._infer_flat_dim(input_shape)
        
        self.optimizer = self._configure_optimizer(opt_type, learning_rate, l2_reg)

        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, fc_units),
            act_fn,
            nn.Dropout(drop_p),
            nn.Linear(fc_units, 10)  # Output layer for 10 classes
        )

    def _infer_flat_dim(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.feature_extractor(dummy_input)
        return out.view(out.size(0), -1).size(1)

    def _configure_optimizer(self, opt_type, lr, weight_decay):
        optimizers = {
            'adam': lambda p: torch.optim.Adam(p, lr=lr, weight_decay=weight_decay),
            'sgd': lambda p: torch.optim.SGD(p, lr=lr, weight_decay=weight_decay, momentum=0.9),
            'rmsprop': lambda p: torch.optim.RMSprop(p, lr=lr, weight_decay=weight_decay),
            'nadam': lambda p: torch.optim.NAdam(p, lr=lr, weight_decay=weight_decay),
        }
        return optimizers[opt_type](self.parameters())

    def _build_conv_layers(self, in_channels, channel_list, kernel_list, use_bn, act_fn):
        layers = []
        for out_channels, k_size in zip(channel_list, kernel_list):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act_fn)
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def training_model(self, train_loader, val_loader, epochs, device, log_fn=lambda m: None):
        """
        Runs the training loop with validation per epoch. Supports AMP (automatic mixed precision).
        """
        self.to(device)
        scaler = torch.cuda.amp.GradScaler()

        for ep in range(1, epochs + 1):
            self.train()
            epoch_loss, correct_preds, total_seen = 0, 0, 0

            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self(imgs)
                    loss = self.loss_fn(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()
                correct_preds += (outputs.argmax(1) == targets).sum().item()
                total_seen += targets.size(0)

            train_acc = 100 * correct_preds / total_seen
            train_loss = epoch_loss / len(train_loader)

            # Validation phase
            self.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    preds = self(imgs)
                    loss = self.loss_fn(preds, targets)

                    val_loss += loss.item()
                    val_correct += (preds.argmax(1) == targets).sum().item()
                    val_total += targets.size(0)

            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {ep}/{epochs} | "
                  f"training Loss: {train_loss:.4f}, training accuracy: {train_acc:.2f}% | "
                  f"validation Loss: {avg_val_loss:.4f}, validation accuracy Acc: {val_acc:.2f}%")

            log_fn({
                'validation_loss': avg_val_loss,
                'validation_accuracy': val_acc,
                'epoch': ep,
                'train_accuracy': train_acc,
                'train_loss': train_loss,
            })

        torch.cuda.empty_cache()
        
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb_entity", "-we", default="cs24m020")
    parser.add_argument("--wandb_project", "-wp", default="dl_a2_part1")
    parser.add_argument("--batch_norm", "-bn", choices=["true", "false"], default="true")
    parser.add_argument("--batch_aug", "-bn", choices=["true", "false"], default="true")
    parser.add_argument("--dense_layer", "-dl", type=int, default=128)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", "-w_d", type=float, default=0.0)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--dropout", "-dp", type=float, default=0.2)
    parser.add_argument("--activation", "-a", choices=['relu', 'elu', 'silu'], default="relu")
    parser.add_argument("--optimizer", "-o", choices=['nadam', 'adam', 'rmsprop'], default="rmsprop")
    parser.add_argument("--activation", "-a", choices=['relu', 'elu', 'selu', 'silu'], default="relu")
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K")
    parser.add_argument("--num_filters", "-nf", nargs=5, type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--filter_sizes", "-fs", nargs=5, type=int, default=[5, 5, 5, 5, 5])
    
    
    return parser.parse_args()

def main():
    args = parse_args()

    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    dataset_dir = 'inaturalist_12K'
    
    # Check if the directory already exists
    if not os.path.exists(dataset_dir):
        # If it doesn't exist, download and extract the dataset
        os.system("wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip -O nature_12K.zip")
        os.system("unzip -q nature_12K.zip -d {}".format(dataset_dir))  # Unzip to the desired directory
        os.system("rm nature_12K.zip")  # Remove the zip file after extraction
        print(f"Dataset downloaded and extracted to {dataset_dir}")
    else:
        print(f"Dataset already exists at {dataset_dir}")

    

    act_fn_map = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'silu': nn.SiLU()
    }

    act_fn = act_fn_map[args.activation]

    # Loader setup
    loader = INatSplitLoader(
        train_root=f"{args.base_dir}/train",
        test_root=f"{args.base_dir}/val",
        img_size=(256, 256),
        batch_size=args.batch_size,
        num_workers=2
    )
    loader.build()
    train_dl, val_dl, _ = loader.get_loaders()

    # CNN setup
    cnn = INatCustomCNN(
        input_shape=(3, 256, 256),
        conv_channels=args.num_filters,
        kernel_sizes=args.filter_sizes,
        fc_units=args.dense_layer,
        act_fn=act_fn,
        use_batchnorm=(args.batch_norm.lower() == "true"),
        drop_p=args.dropout,
        opt_type=args.optimizer,
        learning_rate=args.learning_rate,
        l2_reg=args.weight_decay
    )

    # Training
    cnn.training_model(
        train_loader=train_dl,
        val_loader=val_dl,
        epochs=args.epochs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        log_fn=wandb.log
    )

    wandb.finish()

if __name__ == "__main__":
    main()
