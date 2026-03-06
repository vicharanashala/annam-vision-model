# SAM 3 (Segment Anything Model 3) Results

**Model Description:** The latest and most stable architecture used in the project. Extensive experiments were conducted on dataset preprocessing (mixing Normal, Segmented, and Grayscale data) to bridge the gap between synthetic lab data and real world conditions. 

### Performance Metrics

| Dataset Used | Classes | Validation Accuracy | Testing Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Grayscale PlantVillage + Seg PlantDoc | 10 (Apple/Grapes/Corn/Tomato) | 86.63% | **85.26%** | **Final stabilized architecture.** Prevented background color confusion. |
| Segmented PlantVillage + Seg PlantDoc | 5 Classes 
(Apple/Grapes) | 82.23% | 87.75% | Highly effective for specific datasets. |
| Grayscale PlantVillage + Seg PlantDoc | 5 Classes 
(Apple/Grapes) | 86.00% | 89.79% | Highest individual experiment performance recorded. |
| Segmented Datasets | 5 (Apple/Grapes) | 86.00% | 83.66% | Strong baseline for fully segmented pipelines. |
| Segmented PlantVillage + Seg PlantDoc | 5 (Corn/Tomato) | 86.00% | 82.60% | Consistent performance on specific crops. |
| Normal PlantVillage + Seg PlantDoc | 10 (Apple/Grapes/Corn/Tomato) | 85.56% | 81.00% | Good validation, but Grayscale mix proved superior for testing. |
| Segmented PlantVillage + Seg PlantDoc | 10 (Apple/Grapes/Corn/Tomato) | 83.42% | 81.00% | Fully segmented approach on complex class load. |
| Normal PlantVillage + Seg PlantDoc | 5 (Corn/Tomato) | 87.15% | 78.26% | |
| Normal PlantVillage + Seg PlantDoc | 4 (Bell/Potato) | 76.00% | 69.23% | Early SAM 3 baseline. |
| Grayscale PlantVillage + Seg PlantDoc | 5 (Corn/Tomato) | 86.00% | 73.91% | Struggled slightly on this specific 5 class split. |
