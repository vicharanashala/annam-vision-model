## 12/02/2026
Trained the using ResNet50 model but accuracy dropped from 0.93-0.94 to 0.9.
While going through the top performers kaggle notebook found that some vulnerabilites were reported in the dataset that it had same images with different labels.
Switched to plantvillage and plantdoc dataset trained EfficientNetB0 on plant village dataset got the accuracy of 0.96 but testing with plantdoc dataset it just gave the accuracy of 0.49.
Understood different SAM models and how it's segmenting techniques works. 

## 13/02/2026
Trained the plant village dataset using SAM with mask as a segmentation method and got the accuracy of 0.98 on validation. Then tried different segmentation methods such as cropping, blurring, and removing background. With masking on plant doc the accuracy was 0.78. Then when tested the model on plant doc dataset using cropping it was not able to process all the class due to memory limitations on colab. So, trained 5 classes on CPU of tomato dataset and accuracy was not good it gave around 0.38. So, did some fine tuning and accuracy increased to 0.52

