# Deep Learning Assignment 2: Image Classification on iNaturalist

## Overview
In this assignment, I tackle the image classification challenge on a subset of the iNaturalist dataset using two distinct strategies:
1. **Part A** – Crafting a CNN architecture from the ground up  
2. **Part B** – Leveraging transfer learning by fine‑tuning established ImageNet models  

---

## Repository Layout

### Part A: Home‑grown Convolutional Network
- **Goal**: Implement and train a CNN from scratch without relying on pre‑trained weights.  
- **Highlights**:  
  - Parameterizable layers (activations, regularization, augmentations)  
  - Modular design to easily swap in different filter sizes and depths  
- **Source Code**: [A2_Part_A on GitHub](https://github.com/Vishnu000000/DeepLearning_A2/tree/main/A2_Part_A)

### Part B: Transfer Learning Pipelines
- **Goal**: Boost accuracy by fine‑tuning backbone networks like InceptionV3, ResNet, and Xception.  
- **Highlights**:  
  - Options for partial or full unfreezing of base model layers  
  - Customizable classification head with dropout and weight decay  
- **Source Code**: [A2_Part_B on GitHub](https://github.com/Vishnu000000/DeepLearning_A2/tree/main/A2_Part_B)

---

## Hyperparameter Search Space

### Common Settings
- **Dropout Rates**: 0.0, 0.2, 0.4  
- **Batch Sizes**: 32, 64  
- **Use Data Augmentation**: Yes / No
- **Head Dense Units**: 128, 256, 512 

### Specific to Part A
- **Activation Choices**: ReLU, ELU, SiLU  
- **Filter Configurations**: e.g., [32, 64, 128, 256, 512] at different depths  
- **L2 Penalty**: 0.0, 5e-5, 5e-4
- **Learning Rate**: 1e-4, 1e-3
- **Filter Size**: e.g., [3,3,3,3,3] at different depths 
  

### Specific to Part B
- **Backbone Models**: ResNet, Xception  
 

---

## Detailed Workflow

### Building the Custom CNN
- **Function**: `build_cnn()`  
- **Capabilities**:  
  - Flexible stack of convolutional blocks with configurable filter counts and kernel sizes  
  - Choice of activation, batch normalization, dropout, and L2 regularization  
  - Final fully‑connected layers sized per the number of target classes  

### Transfer Learning Procedure
1. **Load** a pre‑trained feature extractor without its top classifier.  
2. **Attach** a new dense head (fully‑connected layers + dropout).  
3. **Freeze** selected layers of the base network to retain learned features.  
4. **Optionally unfreeze** more layers for deeper fine‑tuning once the head has stabilized.  

---

## Training & Validation

- **Compilation**: Optimizer with an adaptive learning rate (default 1e-4)  
- **Fit Loop**: Train with `model.fit()`, specifying validation split  
- **Augmentations**: Real‑time image transforms (random flip, rotation, etc.) when enabled  

### Evaluation Metrics
- Overall accuracy and loss curves  
- Per‑class precision and recall  
- Batch or full‑set predictions via a `predict()` utility  

---

## Results & Insights

Check out the detailed report on Weights & Biases:  
[Assignment 2 Report on W&B](https://wandb.ai/cs24m022-iit-madras-foundation/Deep_Learning_Assignment2_cs24m022/reports/Assignment-2--VmlldzoxMjM2NjYyNw)

**Key Takeaways**  
- How freezing various layers affects training dynamics  
- The impact of data augmentation on generalization speed  
- Optimal dropout settings to balance overfitting and underfitting  

---

I hope this guide helps you reproduce and extend our experiments. Feel free to open issues or pull requests to share your improvements and findings! 🚀  
