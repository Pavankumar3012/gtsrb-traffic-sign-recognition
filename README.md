# Traffic Sign Recognition with ResNet-18 on GTSRB

This repository contains my CIS 678 Machine Learning project extension, where I fine-tune a ResNet-18 model on the [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) traffic sign dataset and add explainability with Grad-CAM.

## What this project does

- Uses **ResNet-18** pretrained on ImageNet and adapts it to 43 GTSRB traffic sign classes.
- Applies **data augmentation** (rotation, color jitter, random flips) to improve robustness.
- Evaluates the model with accuracy, classification report, and confusion matrix.
- Visualizes **misclassified examples** to understand common error patterns.
- Uses **Grad-CAM** to highlight which parts of each image the model focuses on when predicting.

## Files

- `Project_Implementation.ipynb` – main notebook with:
  - data loading (GTSRB via `torchvision.datasets.GTSRB`)
  - train/validation split
  - ResNet-18 fine-tuning
  - evaluation & confusion matrix
  - misclassified samples viewer
  - Grad-CAM visualizations
- `CIS678_ResNet_Extension_Report.pdf` – written report describing the extension.
- `CIS678_Extended_Project_Slides.pptx` – slides used for the project presentation.

## Dataset

- Dataset: **German Traffic Sign Recognition Benchmark (GTSRB)**.
- The dataset is **not included** in this repository due to size.
- It is automatically downloaded when running the notebook:

```python
from torchvision import datasets

train_full = datasets.GTSRB(root="./gtsrb_data", split="train", download=True, transform=...)
test_full  = datasets.GTSRB(root="./gtsrb_data", split="test",  download=True, transform=...)
