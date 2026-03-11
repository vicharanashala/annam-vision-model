## Datasets

| Dataset Name | Size (Records) | Classes
| :---: | :---: | :---: |
| New Plant Disease Dataset (Augmented) | 87,000 | 38 |
| Plant-Doc | 2,572 | 28 |
| Plant-Wild v1 | 18,542 | 89 |
| Rice Disease Dataset | 45,400 | 19 |

## Results

| Dataset | Transformations Applied | Model Used | Optimizer | Epochs | Testing Accuracy |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **New Plant Diseases + Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 20 | **57.72%** |
| **New Plant Diseases + Plant_Doc + Plant_Wild** | Resized to 380, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 20 | **54.51%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001 for 20 epochs, 0.00001 for 20) | 40 | **55.58%** |
| **Plant_Doc + Plant_Wild** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B7 | AdamW (lr = 0.0001 for 20 epochs, 0.00005 for 20) | 40 | **52.85%** |
| **Plant_Doc + Plant_Wild [Potato Only Subclasses]** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 20 | **69.29%** |
| **Plant_Doc + Plant_Wild [Tomato Only Subclasses]** | Resized to 224, YOLO v11 pipeline cropping, RandomHorizontalFlip | Efficient Net B7 | AdamW (lr = 0.0001) | 20 | **59.86%** |
| **Rice Disease Dataset** | Resized to 224, Rotation (0-180°), RandomHorizontalFlip | Efficient Net B4 | AdamW (lr = 0.0001) | 10/15/20 | **87.10%/87.31%/89.68%** |

## Notes
- Testing for all the models trained on Plant_Doc/Plant_Wild/New_Plant_Diseases or a combination of these datasets is done for 10 classes for comparison,
    - Potato_Early_blight
    - Potato_healthy
    - Potato_Lateblight
    - Tomato_Bacterial_spot
    - Tomato_Early_blight
    - Tomato_healthy
    - Tomato_Late_blight
    - Tomato_Leaf_mold
    - Tomato_Septoria_leaf_spot
    - Tomato_Tomato_Yellow_Leaf_Curl_Virus

- Testing for the models trained on Plant_Doc+Plant_Wild[27 Classes] is done on the following classes,
    - Apple Scab Leaf
    - Apple leaf
    - Apple rust leaf
    - Bell_pepper leaf
    - Bell_pepper leaf spot
    - Blueberry leaf
    - Cherry leaf
    - Corn Gray leaf spot
    - Corn leaf blight
    - Corn rust leaf
    - Grape leaf
    - Grape leaf black rot
    - Peach leaf
    - Potato leaf early blight
    - Potato leaf late blight
    - Raspberry leaf
    - Soyabean leaf
    - Squash Powdery mildew leaf
    - Strawberry leaf
    - Tomato Early blight leaf
    - Tomato Septoria leaf spot
    - Tomato leaf
    - Tomato leaf bacterial spot
    - Tomato leaf late blight
    - Tomato leaf mosaic virus
    - Tomato leaf yellow virus
    - Tomato mold leaf

- Testing for all the models trained on Rice Disease Dataset is done for all 19 classes.