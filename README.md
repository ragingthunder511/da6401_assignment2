# DA6401 Assignment 2: iNaturalist Image Classification

**Report:**  
https://wandb.ai/karekargrishma1234-iit-madras-/cs24m020_dl_a2_sweep1/reports/CS24M020-DA6401-Assignment-2-Report--VmlldzoxMjM2NjY0OQ

**Repository:**  
https://github.com/ragingthunder511/da6401_assignment2

---

## Table of Contents

- [Introduction](#introduction)  
- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Part A: Custom CNN](#part-a-custom-cnn)  
  - [Part B: Transfer Learning](#part-b-transfer-learning)  
- [Hyperparameters](#hyperparameters)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Introduction

This assignment tackles image classification on a subset of the iNaturalist dataset using two complementary approaches:

1. **Part A**: Design and train a Convolutional Neural Network from scratch in PyTorch Lightning.  
2. **Part B**: Fine‑tune pre‑trained ImageNet backbones (e.g., ResNet50, Xception) via transfer learning.

Each part logs metrics to Weights & Biases for easy comparison and visualization.

---

## Project Structure
- **`da6401_assignment2/`**
  - **`README.md`**
  - **`requirements.txt`**
  - **`cs24m020_dl_a1_partA.ipynb`** – Notebook for Part A (CNN from scratch)
  - **`cs24m020_dl_a1_partB.ipynb`** – Notebook for Part B (transfer learning)
  - **`train_A.py`** – CLI script to run Part A experiments
  - **`train_B.py`** – CLI script to run Part B experiments

---

## Prerequisites

- Python 3.8 or higher  
- GPU with CUDA support (optional but recommended)  
- Git

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ragingthunder511/da6401_assignment2.git
   cd da6401_assignment2

2 **Install Dependencies** - pip install -r requirements.txt

3 **Authenticate with Weights & Biases** - wandb.login

## Configuration

### Common Flags (for both Part A and B)

| Flag            | Default   | Description                                  |
|-----------------|-----------|----------------------------------------------|
| `--data_dir`    | `data/`   | Path to iNaturalist image folders            |
| `--epochs`      | `20`      | Number of training epochs                    |
| `--batch_size`  | `32`      | Samples per batch                            |
| `--learning_rate` | `1e-4`  | Initial learning rate for optimizer          |
| `--optimizer`   | `adam`    | Optimizer: `adam`, `sgd`, `rmsprop`, etc.    |
| `--dropout_rate`| `0.3`     | Dropout probability                          |
| `--weight_decay`| `1e-5`    | L2 regularization coefficient                |

### Additional Flags (only for Part B)

| Flag             | Default    | Description                                         |
|------------------|------------|-----------------------------------------------------|
| `--backbone`      | `resnet50` | Backbone model: `resnet50`, `xception`, `inceptionv3` |
| `--freeze_layers` | `True`     | Whether to freeze backbone weights during training |

## Running Experiments

### Part A: Custom CNN

#### Launch a hyperparameter sweep :
wandb sweep sweep_config.yaml

### Start the Sweep Agent
To begin a hyperparameter sweep using Weights & Biases, run the following command with your entity and sweep ID:
wandb agent YOUR_ENTITY/Assignment2_PartA/SWEEP_ID


### Or Run a Single Experiment Manually
You can bypass sweeps and launch a specific configuration directly

## Part B: Transfer Learning
To fine-tune a pre-trained model (e.g., ResNet50), use the following command:

python fine_tune.py \
  --data_dir data/ \
  --backbone resnet50 \
  --epochs 15 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --dropout_rate 0.4 \
  --freeze_layers True \
  --wandb_project DL_Assignment2_PartB \
  --wandb_entity your_entity

## 📊 Evaluation

- **Accuracy & Loss**: Tracked per epoch on training and validation sets
- **Per-Class Metrics**: Precision and recall to identify challenging classes
- **Confusion Matrix**: Visualize common misclassifications
- **Inference**: Use `predict()` in notebooks or scripts for test data evaluation

---

## 📈 Results & Links

- 🔗 **Part A Dashboard**: [DL_Assignment2_PartA](https://wandb.ai/your_entity/DL_Assignment2_PartA)
- 🔗 **Part B Dashboard**: [DL_Assignment2_PartB](https://wandb.ai/your_entity/DL_Assignment2_PartB)

> Dive into the dashboards to compare different runs and hyperparameter configurations!

---

## 🤝 Contributing
I welcome your ideas and improvements! Here's how to contribute:

1. **Fork this repository**
2. **Create a new feature branch**
3. **Commit your changes**
4. **Push to your fork**
5. Open a Pull Request and describe your changes

Let’s build something awesome together 🚀

