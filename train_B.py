import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import wandb
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class list
classesList = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

# Common transforms
common_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Data loading function
def load_data(data_dir, batch_size):
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    full_dataset = datasets.ImageFolder(train_path, transform=train_transforms)

    # Split 80-20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = common_transforms

    test_dataset = datasets.ImageFolder(val_path, transform=common_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# Accuracy/Eval
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

# Model definition
def build_model(activation='relu', dropout=0.2, dense_layer=128, batch_norm=True):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    activation_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

    layers = [nn.Dropout(dropout), nn.Linear(model.fc.in_features, dense_layer)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(dense_layer))
    layers += [activation_fn, nn.Linear(dense_layer, 10)]

    model.fc = nn.Sequential(*layers)
    return model.to(device)

# Main training logic
def main(args):
    # Initialize wandb
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update(args)

    # Load data
    train_loader, val_loader, test_loader = load_data(args.base_dir, args.batch_size)

    # Build model
    model = build_model(activation=args.activation, dropout=args.dropout,
                        dense_layer=args.dense_layer, batch_norm=args.batch_norm == "true")

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.NAdam(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc * 100,
            'val_loss': val_loss,
            'val_accuracy': val_acc * 100
        })

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_acc * 100
    })

    # Save model
    torch.save(model.state_dict(), "finetuned_resnet50.pth")
    wandb.save("finetuned_resnet50.pth")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we", default="cs24m020")
    parser.add_argument("--wandb_project", "-wp", default="dl_a2_partb")
    parser.add_argument("--epochs", "-e", default=10,type=int)
    parser.add_argument("--batch_size", "-b",default=32, type=int)
    parser.add_argument("--optimizer", "-o", default="rmsprop",choices=["adam", "nadam", "rmsprop"] )
    parser.add_argument("--learning_rate", "-lr",default=0.0001, type=float)
    parser.add_argument("--weight_decay", "-w_d", default=0.0, type=float)
    parser.add_argument("--activation", "-a", default="relu",choices=["relu"] )
    parser.add_argument("--dense_layer", "-dl", default=256, type=int)
    parser.add_argument("--dropout", "-dp", type=float, default=0.2)
    parser.add_argument("--batch_norm", "-bn", choices=["true", "false"], default="true")
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K")

    args = parser.parse_args()
    main(args)



