## 12/02/2026
Trained the using ResNet50 model but accuracy dropped from 0.93-0.94 to 0.9.
While going through the top performers kaggle notebook found that some vulnerabilites were reported in the dataset that it had same images with different labels.
Switched to plantvillage and plantdoc dataset trained EfficientNetB0 on plant village dataset got the accuracy of 0.96 but testing with plantdoc dataset it just gave the accuracy of 0.49.
Understood different SAM models and how it's segmenting techniques works. 

## 13/02/2026
Trained the plant village dataset using SAM with mask as a segmentation method and got the accuracy of 0.98 on validation. Then tried different segmentation methods such as cropping, blurring, and removing background. With masking on plant doc the accuracy was 0.78. Then when tested the model on plant doc dataset using cropping it was not able to process all the class due to memory limitations on colab. So, trained 5 classes on CPU of tomato dataset and accuracy was not good it gave around 0.38. So, did some fine tuning and accuracy increased to 0.52

## 14/02/2026
Initially added 5 more classes by doing the segmentation on it. In which 3 classes were of apple and 2 of bell. Then trained using total 10 classes but accuracy didn't improve much. Then again added 5 more classes in which 3 is of corn and 2 of potato. Model is under training.

## 16/02/2026
Even after increasing the total number of classes to 15 accuracy wasn't improving much. So, went through the segmentated images again and realized that images are inconsistently cropped some images are big while some were small. Hence, changed the segmentation method from single center point segmentation to bounding box and did the preprocessing on 7 classes using this method. After training it gave a validation score of around 0.65 which was some good improvement from 0.52.
