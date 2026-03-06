# SAM 1 (Segment Anything Model 1) Results

**Model Description:** The first iteration of integrating instance segmentation to isolate diseased leaf areas. Initial attempts used single center-point cropping, which was later upgraded to a Bounding Box method to ensure uniform feature extraction.
For the segmentation SAM 1 is used and for classification EfficientNet B3.
### Performance Metrics

| Dataset Used | Classes | Validation Accuracy | Testing Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Segmented Data | 15 (Using Eff. B3) | 39.00% | N/A | Center-point cropping caused inconsistent image sizes. |
| PlantDoc Only (Bounding Box) | 5 (Apple/Bell) | 64.86% | N/A | Bounding box method immediately improved validation. |
| Train PlantVillage, Fine-tune PlantDoc | 4 (Bell/Potato) | 51.85% | N/A | Fine-tuning alone was insufficient. |
| Normal PlantVillage + SAM 1 Seg PlantDoc | 4 (Bell/Potato) | 85.19% | 58.00% | Merging datasets boosted validation, but testing remained low. |
