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

## 18/02/2026
- Trained Efficient Net B4 for 20 Epochs on the New_Plant_Disease+Plant_Doc+Plant_Wild dataset.
- Images were cropped and resized to 224x224.
- On Testing Set,
    - Accuracy => 57.08%
- Tried to change the image size to 380 but that decreased the accuracy.
- Tried Efficient Net B4 with only Plant_Doc+Plant_Wild dataset.
- On Testing Set,
    - Accuracy => 53%
- Abandoned the Tomato_Target_Spot as well as Tomato_Spider_mites_Two-spotted_spider_mites classes as of now, since the training data associated
  with these classes has not been enough for the model to capture features from them properly.
- Tried the various versions of the upgraded EfficientNetV2(small, medium, large) but all of them stalled at 50-52% testing accuracy.

## 19/02/2026
- Currently the best accuracy on the test set of Plant_Doc+Plant_Wild is 55.57% using an Efficient Net B4 model trained for 40 epochs on the cropped images of Plant_Doc+Plant_Wild
- Trying to pretrain InceptionResnetV2 using New Plant Diseases(Augmented) dataset and then finetuning it on the merged Plant_Doc+Plant_Wild dataset. Anything close to a 70% test accuracy can be considered a success for this case.

## 23/02/2026
- Tested EfficientNetB4 for Plant_Doc + Plant_Wild dataset for Potato only,
    Testing Accuracy => 69.29 %
- Tested EfficientNetB7 for Plant_Doc + Plant_Wild dataset for Tomato only,
    Testing Accuracy => 58.72 %
- Tested ConvexNet-Base for Plant_Doc + Plant_Wild dataset for Tomato only,
    Testing Accuracy => 66.50 %
- Tested ConvexNet-Base for Plant_Doc + Plant_Wild dataset for Potato only,
    Testing Accuracy => 71.26 %
- Tested ConvexNet-Large for Plant_Doc + Plant_Wild dataset,
    Testing Accuracy => 56.26 %

## 24/02/2026
- Tested EfficientNetB4 for Potato + Tomato Dataset(provided),
    Validation Accuracy => 97.84 %
    Testing Accuracy(On 11 classes) => 37.7 %
- Currently training ConvexNet-Base on the same dataset for 10 epochs.
- Went through the following research papers in search of alternative methods to
  approach the given problem,
    - [Paper 1](https://www.nature.com/articles/s41598-025-92143-0)
    - [Paper 2](https://www.ijert.org/detection-and-classification-of-plant-leaf-diseases-by-using-deep-learning-algorithm)
- The major problem with these research papers is that they have used lab-controlled 
  images for the training and testing part. The models trained by me worked on these datasets well but 
  they struggled on real-world images.

## 25/02/2026
- Tested ConvexNet-Base for Potato+Tomato Dataset(provided),
    - Validation Accuracy => 97.78 %
    - Testing Accuracy(On 11 classes) => 38.52 %
- Researching about various transformations available in PyTorch and if some 
  particular transformations can help us cover up the deficit in our training 
  data quantity.
- After applying transforms.equalize() on the Plant_Doc+Plant_Wild(Potato Only) dataset(used ConvexNet-Base only), saw a jump in validation as well as testing accuracy;
    - Validation Accuracy => 75.68 %
    - Testing Accuracy => 73.62 %
- Applied the same transformation on the complete dataset and it bumped the 
  testing accuracy from 56.26 % to 59.63 %
- Further testing with other parameters is required before making a conclusion.

## 26/02/2026
- Applied Photometric transformation (transforms.ColorJitter including saturation and hue) as well as Affine Geometric transformations yet no change in accuracy, instead it fell down by 1-3 %.
- Best result till date,
    - Model: ConvexNet-Base
    - Dataset: Plant_Doc+Plant_Wild
    - Epochs: 20
    - Testing Accuracy[On the 10 classes]: 60.21 %
- Currently compiling a jupyter notebook containing the saved models along with their class-wise classification accuracy on the testing set.

## 27/02/2026
- Compiled and pushed the jupyter notebook containing the test results from various models on the testing set provided for the Potato and Tomato subclass.
- Initialized training on the Rice Disease Dataset with some basic transformations applied and EfficientNetB4 model for 10 epochs,
    - Validation Accuracy => 92.47 %
    - Testing Accuracy => 87.10 %
- Next step, would be try to train this model for another 5 epochs and verify it's validation accuracy and find the overfit spot.
- Also, training on the Convex-Net-Base after this to compare performances.

## 28/02/2026
- Compiled the monthly update with all the model tested along with the datasets and the testing accuracies.
- Trained and tested ConvexNet-Base on Rice Diseases Dataset with basic transformations for 20 epochs,
    - Validation Accuracy => 93.21 %
    - Testing Accuracy => 94.19 %
- Trained and tested ConvexNet-Base on Rice Diseases Dataset with Equalize transformation for 20 epochs,
    - Validation Accuracy => 92.42 %
    - Testing Accuracy => 91.40 %
- Currently training a ConvexNet-Base after applying a CenterCrop and removing Equalize transformation.

## 02/03/2026
- Trained ConvexNet-Base after applying Center Crop and removing Equalize for 20 epochs,
    - Validation Accuracy => 93.69 %
    - Testing Accuracy => 88.39 %
- Trained a ConvexNet-Large as well for 15 epochs to verify the performance of bigger models,
    - Validation Accuracy => 93.31 %
    - Testing Accuracy => 87.74 %
- Implemented a WeightedRandomSampler on the training data loader as to focus more on the classes with very less number of training examples. Currently training this using ConvexNet-Base.

## 05/03/2026
- Trained a ConvexNet-Base after applying WeightedRandomSampling with replacement on the train loader for 20 epochs,
    - Validation Accuracy => 88.82 %
    - Testing Accuracy => 91.40 %
- Trained a ConvexNet-Base after applying WeightedRandomSampling with replacement on the train loader for 10 epochs,
    - Validation Accuracy => 90.19 %
    - Testing Accuracy => 93.98 %
- Compiled a first draft for the results.md file and pushed it to the repository.
- Currently looking for alternative models that can work better with the Rice Disease Dataset, InceptionV3 and VGG16 are 2 potential candidates (training InceptionV3 as of now).

## 06/03/2026
- Trained a InceptionV3 model after resizing the images to (299,299) on the Rice Disease Dataset for 30 epochs,
    - Validation Accuracy => 93.01 %
    - Testing Accuracy => 93.12 %
- Trained a InceptionV3 model after resizing the images to (299,299) and applying WeightedRandomSampling with replacement on the train loader of the Rice Disease Dataset for 30 epochs,
    - Validation Accuracy => 90.73 %
    - Testing Accuracy => 92.04 %
- Trained a VGG16 model after resizing the images to (299,299) and applying WeightedRandomSampling with replacement on the train loader of the Rice Disease Dataset for 30 epochs,
    - Validation Accuracy => 87.01 %
    - Testing Accuracy => 90.11 %
- Trained a VGG16 model after resizing the images to (224,224) and applying WeightedRandomSampling with replacement on the train loader of the Rice Disease Dataset for 30 epochs,
    - Validation Accuracy => 87.59 %
    - Testing Accuracy => 89.68 %
- Currently training a VGG16 model for Images resized to (224,224) but no sampling applied for 20 epochs.

## 07/03/2026
- Trained a VGG16 model after resizing the images to (224,224) of the Rice Disease Dataset for 30 epochs,
    - Validation Accuracy => 88.99 %
    - Testing Accuracy => 83.66 %
- Trained a Convex-Net-Base model after resizing the images to (224,224) and replacing the images of the Brown Spot class of the Rice Disease Dataset for 20 epochs,
    - Validation Accuracy => 94.44 %
    - Testing Accuracy => 86.67 %

## 09/03/2026
- After learning the correct transformation and settling on ConvexNet-Base as the choice of model, trained ConvexNet-Base on the Merged Dataset of Plant_Doc+Plant_Wild (27 classes) for 20 epochs after resizing the images to 224x224 and applying histogram equalization;
    - Validation Accuracy => 68.54 %
    - Testing Accuracy => 73.73 %
- Also, trained YOLO26 on the same dataset with no transformations for 100 epochs with a patience=30(i.e. stop training if no change in accuracy for 30 epochs),
    - Validation Accuracy => 62.9 %
    - Testing Accuracy => 64.2 %
- The images used for training both of the above-mentioned models were first passed through YOLO v11 pipeline to crop the images to focus on a leaf.
- Currently exploring transformations described under Ultralytics documentation that can be applied on the YOLO26 pipeline.

## 10/03/2026
- Trained YOLO26-Large on the cropped and merged Plant_Doc+Plant_Wild dataset[27 classes] for 100 epochs after applying histogram equalization, horizontal flipping and normalization;
    - Validation Accuracy => 65.7 %
    - Testing Accuracy => 72.5 %
- Trained YOLO26-X on the cropped and merged Plant_Doc+Plant_Wild dataset[27 classes] for 100 epochs after applying histogram equalization, horizontal flipping and normalization;
    - Validation Accuracy => 64.9 %
    - Testing Accuracy => 70.3 %
- Currently training YOLO26-Large on the Rice Disease Dataset for 100 epochs.

## 12/03/2026
- Compiled all the necessary inferences as well as the class-wise accuracy in the results.md file and compilation_of_models_testing.ipynb respectively.
- Implemented Grad-Cam for Potato and Tomato Dataset as well as Rice Disease Dataset for further interpretability of results and decisions made by the model.
- Currently researching about VLMs, their architecture and how to implement them to solve the objective given to us.

## 13/03/2026
- Researching VLM in context to image classification, currently went through the following articles:
    - [Benchmarking Top Vision Language Models (VLMs) for Image Classification](https://www.clarifai.com/blog/best-vision-language-models-vlms-for-image-classification-performance-benchmarks)
    - [Why are Visually-Grounded Language Models Bad at Image Classification?](https://arxiv.org/html/2405.18415v2)
- Implemented Grad-Cam for both disease datasets, now working on expanding the plotting grid as well to include more examples from different classes for inference purposes.
- Shortlisted one model on HuggingFace as well from the above-mentioned articles, "Qwen/Qwen2.5-VL-7B-Instruct".

## 16/03/2026
- Created the .jsonl file but experiencing issues storing the images on the VM. Also, the repeated crashing of VM interrupts the loading of the VLM.
- Pushed the Grad-CAM images along with the results as a Jupyter notebook titled "compilation_of_models_testing_dataset".
- Currently focusing on setting up the VLM pipeline as of now.

## 17/03/2026
- Created the VLM pipeline for Qwen-2.5-7B-Instruct, trained the model for 100 steps as well but encountered bugs during inference so currently working on that as of now.

## 18/03/2026
- Trained the VLM pipeline for 200-300 steps which took 2 hours and 40 mins to execute completely. The issue was: it was not training over the complete dataset so changed the training strategy to epochs and started training it for 1 epoch.
- Training it for 1 epoch with Unsloth and PEFT using LoRA adapters gave an estimated time of 16 hours which is not feasible on the Kaggle GPU quota as of this week.
- Researching about various ways to finetune this model so as to ensure what should I expect in terms of training times and if I can shave down the training time from 16 hours to 7-8 hours.

## 19/03/2026
- Created a pipeline with Qwen-3-2B but the training time per epoch was infeasible (32 hours per epoch), reduced the image size as well increased the batch size but it made no difference.
- Currently working on a Qwen-2 pipeline with Unsloth to train the model even if it takes 16 hour per epoch (Will save checkpoints to resume training) as that seems more feasible to train and test as of now.

## 20/03/2026
- Came across several issues with the created pipeline, the major one was the vector size mismatch.
- Reduced the size of the images of 252x252 to 224x224 to ensure correct calculations but that did not help with the problem. Confined the LoRA parameters to language layers instead of vision+language layers, so far no issues yet.
- As soon as the training is done, will update this document and results.md with the inference results.
- Currently training over the Rice Disease Dataset for 2 epochs using Qwen2-2B-Instruct VLM.

## 23/03/2026
- Training finished for Qwen2-2B-Instruct for **6000 steps**, 
    **Testing Accuracy => 61.52 %** 
  Training the model for another **2000 steps**, to verify whether the model is overfitting or not.
- Training for another **2000 steps** was successful, Testing Accuracy increased from **61.51 % => 63.66 %** 
- Trained the model for another **3,358 steps** on the same **Rice Disease Dataset** before the Kaggle GPU Quota ran out **Testing Accuracy** increased from **63.66 % to 65.38 %**
- Created the pipeline for **Qwen 3-4B-Instruct** in a new Kaggle notebook and currently it is training for **an epoch** or **5679 steps** on the **Rice Disease Dataset**.

## 24/03/2026
- Training finished for **Qwen 3-4B-Instruct** for **5679 steps** or **1 epoch**,
    **Testing Accuracy => 69.03 %**
- Trained the model for another 2 epochs or for a total of **17037 steps**,
    **Testing Accuracy** increased from **69.03% -> 81.08%**
- Currently training for another epoch to verify whether the model will overfit or whether the generalization accuracy can be improved further.

## 25/03/2026
- Kaggle GPU Quota ran out for both of the accounts, so could not proceed with any model training.
- Instead, went back to the codebase and exploring it in detail to find hyperparameters that can be tuned as soon as VM access is provided to improve accuracy.
- Currently have gone through the processor, model and LoRA config in the codebase. Making my way through the data collator as of now.
- As of now, Image Resolution in the Processor as well as rank of LoRA matrix are the two things that can be tweaked for better performance.

## 26/03/2026
- Created a new Kaggle account for further experimentation, trained Qwen 3-4B-Instruct for **5 epochs**.
- Testing Accuracy decreased from **81.08 %** to **80.65 %**
- The model is showing signs of overfitting, considering the training only focused on weights related to the language layers and no vision layers were included due to Kaggle's VRAM limitation.
- Currently training it for 4 epochs, to verify whether some improvement can be observed before the test accuracy started going down for 5 epochs.
- Also, currently looking into **Kimi-VLM** as a potential choice for this task.

## 27/03/2026
- Trained **Qwen 3-4B-Instruct** for **4 epochs** to no avail as the testing accuracy remained stagnant and started decreasing after the **5th epoch**.
- Testing Accuracy for **4 epochs** remained the same at **81.08 %**
- Started working on a Kimi-VLM pipeline but currently it is not working because of dependency issues, tensor dimensions mismatch.
- Fixed the dependency issues but the GPU quota for all 3 accounts ran out, so had to stop the experimentation.


