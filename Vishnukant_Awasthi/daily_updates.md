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

## 17/02/2026
Trained the model on the Plant Village dataset and fine tuned it on PlantDoc, then validated it on the PlantDoc dataset. Only fine tuning on the PlantDoc didn't give much improved accuracy, so also included the PlantDoc in the training dataset. So, merged the plant village and plant doc for training and validating. Plant Village dataset is big as compared to the PlantDoc real world images, in combined validation, 0.98 accuracy was achieved, but there were fewer images in the validation of the real world PlantDoc. So, can't rely on this accuracy for real world images. Hence validated only on plant doc dataset, achieving the accuracy of 0.8518.

## 18/02/2026
Segmented test data for testing accuracy. Completed testing the model on test dataset and testing accuracy is 60%. Trained again through dropout and tested, getting 0.64 accuracy. Then switched to SAM2 from SAM1. Completed the segmentation using SAM 2 large model.

## 19/02/2026
Trained SAM 2 only on plant doc dataset to compare the accuracy with SAM 1. It performed better than SAM 1 with validation accuracy of around 63% and testing accuracy of 59 where as SAM 1 initially just gave 38% accuracy. Then increase the dataset by merging it with plant village dataset and then again trained on merged dataset and validated only on plant doc but testing accuracy remained same 59& while validation accuracy increase from 63% to 68%. Then did the segmentation using different method initially removed the background and made it white. Now, doing the cropping using bounding box.

## 20/02/2026
Trained and tested on the new cropped segmented dataset of SAM 2 large model but its giving the same accuracy of 59%. Changed the SAM 2 model from large to base. Then did the segmentation again using SAM 2 base model. Completed the training and testing and it's giving validation accuracy of 74% and testing accuracy improved from 59% to 69%. Then doing the segmentation on more classes to increase the number of classes from 4 to 13. Adding 9 classes of tomato.

## 21/02/2026
Completed the segmentation on 9 new classes. Trained the model on 13 classes and it gave overall testing accuracy is 51% but for some class it is giving good accuracy like 92%. Explored and learned how to set up and use SAM 3. 

## 22/02/2026 
Segmented 4 classes of plantdoc dataset using SAM 3 model. Also did some finding for the plant doc segmentation annotation dataset. Completed the training and segmentation on test dataset for testing. Completed the testing and it's giving accuracy of 69.23% and validation accuracy of 76.34%.

