# EfficientNet Results

**Model Description:** Standard image classification baseline models (EfficientNet B0 and B3) trained without instance segmentation. These models were used to establish initial benchmarks but struggled significantly with the domain gap between clean lab images and messy real-world photos.

### Performance Metrics

| Dataset Used | Classes | Validation Accuracy | Testing Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| PlantVillage | All | 99.67% (B3) | 25.68% (PlantDoc) | Massive overfitting to clean lab backgrounds. |
| PlantVillage | All | 96.00% (B0) | 49.00% (PlantDoc) | Performed slightly better on real world data but still unreliable. |
