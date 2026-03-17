# Model Results

# Dataset

Dataset: wheat-leaf-disease

Classes:
- Brown rust
- Healthy
- Loose Smut
- Septoria
- Yellow rust

Total Images: 5521

Dataset Split:
- Train: 3864
- Validation: 828
- Test: 829

---

# Hardware

GPU: Tesla T4  
VRAM: 15.6 GB

---

# EfficientNet Experiments

Multiple EfficientNet architectures were evaluated.

Training strategy:
- Transfer learning
- Frozen backbone initially
- Backbone unfrozen after epoch 10
- Adam optimizer
- Learning rate scheduler used

---

# EfficientNet-B0

Parameters:
- Total parameters: ~4.3M
- Trainable parameters (initial): ~0.33M

Example training progress:

Epoch 1  
Train Acc: 0.7156  
Val Acc: 0.8116

Epoch 2  
Train Acc: 0.7808  
Val Acc: 0.8502

Best validation accuracy improved progressively during training.

---

# EfficientNet-B1

Model trained using input size **240**.

Architecture deeper than B0 with improved feature representation.

Validation accuracy improved compared to baseline in later epochs.

---

# EfficientNet-B2

Model trained using input size **260**.

Larger architecture with increased capacity for feature extraction.

Training showed stable convergence and improved validation performance.

---

# EfficientNet-B3

Largest EfficientNet model tested in this experiment.

After backbone unfreezing at epoch 10, validation accuracy improved significantly.

Example improvement:

Epoch 10  
Train Acc: 0.8716  
Val Acc: 0.9191

---

# EfficientNetV2-S Experiment

Separate experiment using EfficientNetV2-S.

Model parameters:
- Total parameters: ~20M

Training setup:
- Head-only training initially
- Backbone unfreezing later
- Class weights applied

Dataset used:
- Brown rust
- Healthy
- Loose Smut
- Septoria
- Yellow rust

Total images: 5521

---

# Observations

- EfficientNet models perform well on wheat disease classification.
- Larger models (B2/B3) show better feature extraction capability.
- Backbone unfreezing significantly improves performance.
- EfficientNetV2-S provides a strong architecture for future experiments.

---

# Conclusion

EfficientNet architectures demonstrate strong performance on wheat disease classification.  
EfficientNet-B0 provides a lightweight baseline, while deeper models such as **B2 and B3 achieve improved validation accuracy**.

Further experimentation will include:
- extended training
- hyperparameter tuning
- comparison with EfficientNetV2 architectures.
