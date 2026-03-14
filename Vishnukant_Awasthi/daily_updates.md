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

## 23/02/2026 
Segmented 4 classes of plantdoc dataset using SAM 3 model. Also did some finding for the plant doc segmentation annotation dataset. Completed the training and segmentation on test dataset for testing. Completed the testing and it's giving accuracy of 69.23% and validation accuracy of 76.34%.

## 24/02/2026 
Explored the SAM 3 model, how the accuracy can be improved further. Found a project cropscan. Understood the code of app.py, leaf_segmenter.py and sam2_segmentation.py of the project. Understood the complete working of the Cropscan project. It does not classify multiple diseases. It only classifies whether a leaf is healthy or not healthy, if not healthy, then it predicts the severity level mild, moderate, or high. Then tested using the weights provided in the project and it gave a testing accuracy of 89.23% for 2 class label, healthy or diseased.

## 25/02/2026 
Segmented 5 classes of plantdoc dataset 3 of Apple and 2 of Grapes using SAM 3 model. Completed the training and segmentation on test dataset for testing. Completed the testing and it's giving testing accuracy of 83.66% and validation accuracy of 86%.

## 26/02/2026 
Explored different dataset on kaggle for better training and testing. Earlier used to train the model on normal plantvillage + segmented plantdoc. Now, trained the model using segmented plantvillage and segmented plantdoc. Completed training and testing. Got the validation accuracy of 82.23% and testing accuracy of 87.75% when both the datasets are segmented. Then trained the model using grayscale plantvillage + segmented plantdoc. Completed the training and testing on Grayscale Plantvillage + Segmentaed Plactdoc. Got the validation accuracy of 86% and testing accuracy of 89.79%.

## 27/02/2026 
Segmented 5 classes of plantdoc dataset 3 of Corn and 2 of Tomato using SAM 3 model. Completed the training and testing for 3 classes of corn and 2 of tomato. For segmented plant village + segmented plant doc got validation accuracy of 86% and testing accuracy of 82.6%. For grayscale plantvillage + segmented plantdoc got validation accuracy of 86% and testing accuracy of 73.91%. For normal plantvillage + segmented plantdoc got validation accuracy of 87.15% and testing accuracy of 78.26%.

## 28/02/2026 
Trained the model on 10 classes 3 of corn, 2 of tomato, 3 of apple and 2 of grapes. For segmented plant village + segmented plant doc got validation accuracy of 83.42% and testing accuracy of 81%. For grayscale plantvillage + segmented plantdoc got validation accuracy of 86.63% and testing accuracy of 85.26%. For normal plantvillage + segmented plantdoc got validation accuracy of 85.56% and testing accuracy of 81%.

## 02/03/2026 
Gone through the different versions and working of DINO. Trained the model using DINOv2 small on 10 classes and the dataset used was segmented. Got the validation accuracy of 80% and testing accuracy of 76%.

## 03/03/2026 
Experimented further with the DINOv2 small model to improve performance. First, fine-tuned the model by unfreezing the last 2 transformer blocks, which gave the validation accuracy of 89% and testing accuracy of around 90.5%. After that, tried unfreezing the last 4 blocks to see if deeper fine-tuning would improve performance. Got the validation accuracy of 89% and testing accuracy of 88.4%. So, unfreezing the last 2 blocks gives better results than unfreezing 4 blocks.

## 05/03/2026 
Trained the model with Weight random sampler and also changed loss from cross entropy to focal loss for handling class imbalance and better accuracy using DINOv2 small. Completed the training using weighted random sampler and got the testing accuracy of 86%. Also collected real world images from different websites to increase the test dataset size. Increased the number of images from 95 to 179. Training the DINOv2 small with 5 folds cross validation. Completed the code setup and started training. Completed training on 4 folds of cross validation. Also tested on new test dataset it's giving 89% accuracy on 179 test images.

## 06/03/2026 
Changed the DINOv2 version from small to base. Trained the model on both the datasets plantvillage and plantdoc segmented. Got the validation accuracy of 88.23 and testing accuray of 91.5 on 95 test images from plantdoc and 82 images collected from different websites so total of 177 testing images. Then trained the model on merged dataset of grayscale plantvillage + segmented plantdoc. Got the validation accuracy of 87.7% and testing accuracy of 88.7%.

## 07/03/2026 
Trained the DINOv2 base model with merged dataset of Normal plantvillage + Segmented plantdoc. Got the validation accuracy of 86%. Completed testing and got the testing accuracy of 90%. Set up the code for normal plantvillage + normal plantdoc for training on this merged dataset without segmentation. Completed the training and got the validation accuracy of 91.44%. Then did the testing and got the testing accuracy of 92.09%. Hence, base model performs better on normal without segmented dataset while small version performed better on the segmented dataset.

## 09/03/2026 
Changed the DINOv2 version from base to large. Set up the code for DINOv2 large model. Started training the model on merged dataset of normal plantvillage + normal plantdoc both were without segmentation. Completed training using DINOv2 large with 3 blocks unfreeze. Got the validation accuracy of 90.9% and testing accuracy of 84.21%. As the model was overfitting added the dropout of 0.5, weighted random sampler and better augmentation and started training the DINOv2 large model. Completed training got the validation accuracy of 90.37% and testing accuracy of 89.24%. Started training the model with more parameters changes unfreezed 4 blocks and training for 30 epochs. Completed 14 epochs.

## 10/03/2026 
Started training the model with more parameters changes ,unfreezed 4 blocks and training for 30 epochs. Completed training on 30 epochs. Got the validation accuracy of 92% and testing accuracy of 89.2%. Trained the model with merged dataset of Segmented plantvillage + segmented plantdoc using DINOv2 large. Got the validation accuracy of 88.23% and testing accuracy of 88.7%. Trained the model with merged dataset of Grayscale plantvillage + segmented plantdoc using DINOv2 large. Got validation accuracy of 88.23% and testing accuracy of 88.7%. Then started training with some parameter changes in loss function and also changed the image size to 384 and unfreeze 6 layers and training for 30 epochs. Completed 12 epochs.

## 11/03/2026 
Completed 30 epochs of training with some parameter changes in the loss function and also changed the image size to 384 and unfreeze 6 layers. Got validation accuracy of 92% and testing accuracy of 89%. Concluded the DINOv2 large. Gone through how to set up DINOv3 small, base and large versions. Installed the required libraries on the VM Jupyter notebook. Got some errors even after installing libraries, it was not getting imported in the code fixed it. Imported required dataset. Completed the setup of Jupyter notebook to run code on it. Switched the model from DINOv2 large to DINOv3 base. Set up the pipeline to train the model using DINOv3 base on normal without segmented dataset for 30 epochs and started training with 2 blocks unfreeze. Got the validation accuracy of 93%.

## 12/03/2026 
Got the testing accurcy of 88% for 2 blocks unfreeze on DINOv3 base. Set up the pipeline for training with 6 blocks unfreeze and changed loss from cross entropy to focal loss to improve the per class accuracy. Completed the training. Got the validation accuracy of 90% on 6 blocks unfreeze with DINOv3 base model and testing accuracy of 90%. Testing accuracy improved 2% from 88% to 90% as compared to the previous model. Setup the pipeline for training on DINOv3 base with 6 blocks unfreeze on normal plantvillage + segmented plantdoc merged dataset for 30 epochs. Completed training on DINOv3 base with 6 blocks unfreeze on normal plantvillage + segmented plantdoc merged dataset for 30 epochs. Got the validation accuracy of 82% and testing accuracy of 90%. Setup the pipeline for training on DINOv3 base with 6 blocks unfreeze on segmented plantvillage + segmented plantdoc merged dataset for 40 epochs. Started training.

## 13/03/2026 
Completed training on DINOv3 base with 6 blocks unfreeze on segmented plantvillage + segmented plantdoc merged dataset for 40 epochs. Got the validation accuracy of 88% and testing accuracy of 90%. Setup the pipeline for training on DINOv3 base with 6 blocks unfreeze on grayscale plantvillage + segmented plantdoc merged dataset for 40 epochs. Completed training on DINOv3 base with 6 blocks unfreeze on grayscale plantvillage + segmented plantdoc merged dataset for 40 epochs. Got the validation accuracy of 82% and testing accuracy of 84%. Changed the model version from DINOv3 base to DINOv3 small. Set up the pipeline to train the model using DINOv3 small on normal without segmented dataset for 40 epochs. Completed training the DINOv3 small on normal + segmented merged dataset with 2 blocks unfreeze got the validation accuracy of 84% and testing accuracy of 86%. Setup the pipeline for training on DINOv3 small with 2 blocks unfreeze on segmented merged dataset of plantvillage and plantdoc for 30 epochs. Completed training on DINOv3 small with 2 blocks unfreeze on segmented plantvillage + segmented plantdoc merged dataset for 30 epochs. Got the validation accuracy of 84% and testing accuracy of 87%.

## 14/03/2026 
Setup the pipeline for training on DINOv3 small with 2 blocks unfreeze on merged dataset of grayscale plantvillage + segmented plantdoc for 30 epochs. Completed training on DINOv3 small with 2 blocks unfreeze on grayscale plantvillage + segmented plantdoc merged dataset for 30 epochs. Got the validation accuracy of 84% and testing accuracy of 83%. Setup the pipeline for training on DINOv3 small with 2 blocks unfreeze on merged dataset of normal without the segmentation dataset for 30 epochs. Completed training on DINOv3 small with 2 blocks unfreeze on normal plantvillage and plantdoc merged dataset without segementation for 30 epochs. Got the validation accuracy of 90% and testing accuracy of 86.57%. Did some parameter changes on image size, loss function and setup the pipeline for training on DINOv3 small with 4 blocks unfreeze on merged dataset of normal without the segmentation dataset for 40 epochs. Started the training.
