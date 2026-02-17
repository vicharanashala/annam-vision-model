# Daily Updates

## 12/02/2026
- Looking for alternative approaches in case this one does not pan out, found an interesting paper on aECA-ResNet34 ([link](https://www.mdpi.com/2073-8994/16/4/451))
- As of now, the accuracy score (except for the Rice Dataset) is 97.9% 
- After training with the Rice Dataset, I will update the score over here.

## 13/02/2026
- Currently using New Plant Disease(Augmented) dataset ([link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download))
- Testing using this dataset's provided test set has resulted in > 90% accuracy but the test dataset shared with us showed < 20%
- One major reason which I could observe for this is the way in which this model was fine-tuned i.e. the training images were that of isolated leaves and not of the whole plant.
- So, the focus has shifted a bit from testing out models to testing out better pre-processing techniques moreover better segmentation techniques to help the model identify the leaf easily.

## 14/02/2026
- Currently training the ViT_l_16 model on the New Plant Disease(Augmented) dataset and then fine-tuning the resultant model on the Plant-Doc dataset ([link](https://github.com/pratikkayal/PlantDoc-Dataset)).
- The goal is to capture the various characteristics of the plant leaves and then try to classify the real world plant disease dataset i.e. Plant-Doc coupled with this knowledge.
- So far, the accuracy of the model has been struggling on the real-world test set.
- Also, unfreezing some of the layers (as much as the GPU allows) to improve the learning and better weights correction of the model.

## 16/02/2026
- Trained Yolo v11 on the Plant-Doc dataset to get bounding boxes, which in turn will be used to train the ViT Classifier.
- The idea is to train the classifier to capture features from the leaves of real world photos and Yolo will help us focus on the leaves of these real world photos.

## 17/02/2026
- Trained and tested a YOLO v11 + ViT_L_16 pipeline on Plant_Doc(28 classes),
    - On Validation Set,
        - Top-1 Accuracy => 47.44%
        - Top-5 Accuracy => 84.44%
    - On Testing Set,
        - Top-1 Accuracy => 41.10%
        - Top-5 Accuracy => 77.97%
    - On Testing Set shared with Us (13 classes),
        - Accuracy => 28%
- Focusing on 11 classes instead of the 13 provided because appropriate data w.r.t. those classes is not available.
- Switching from ViT to a CNN based classifier, currently training and testing them and they are showing promising results on the testing set shared as of now.