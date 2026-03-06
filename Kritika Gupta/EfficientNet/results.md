# Model Results

## Date
06 June 2026

## Dataset Used
- [Wheat Leaf Disease Dataset-1](https://www.kaggle.com/datasets/jayaprakashpondy/wheat-leaf-disease)
- [Wheat Leaf Disease Dataset-2](https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset)

These datasets contain images of plant leaves belonging to different crop disease classes.  
They were used to train a deep learning model for automatic crop disease classification.

---

# Model Architecture
**EfficientNet-B0 (Base Model)**

Transfer learning was applied using a pretrained EfficientNet-B0 model.  
Initially the backbone was frozen and later unfrozen for fine-tuning.

Trainable parameters after unfreezing:
4,336,769 parameters

---

# Training Configuration

|   Parameter   |      Value        |
|---------------|-------------------|
| Framework     | PyTorch           |
| Model         | EfficientNet-B0   |
| Image Size    | 224 × 224         |
| Optimizer     | Adam              |
| Loss Function | CrossEntropyLoss  |
| LR Scheduler  | CosineAnnealingLR |
| Epochs        | 30                |

---

# Training Progress

The model was trained for **30 epochs**.

Example training progress:

Epoch 1  
Train Acc: 71.56%  
Validation Acc: 81.16%

Epoch 10  
Train Acc: 86.08%  
Validation Acc: 89.37%

Epoch 20  
Train Acc: 91.38%  
Validation Acc: 95.17%

Epoch 29  
Train Acc: 92.42%  
Validation Acc: 96.62% (Best)

---

# Final Results

|         Metric           |   Score    |
|--------------------------|------------|
| Best Validation Accuracy | **96.62%** |
| Final Training Accuracy  | **92.11%** |
| Final Validation Loss    | **0.1108** |

---

# Observations

- EfficientNet-B0 performed very well for crop disease classification.
- Fine-tuning the backbone significantly improved performance.
- The model achieved **over 96% validation accuracy**, showing strong ability to distinguish between disease classes.
- Transfer learning helped achieve high accuracy with limited training time.

---

# Conclusion

EfficientNet-B0 proved to be an effective architecture for multi-class crop disease classification using the available datasets. With proper fine-tuning and augmentation, the model achieved **96.62% validation accuracy**, making it suitable for practical crop disease detection systems.
