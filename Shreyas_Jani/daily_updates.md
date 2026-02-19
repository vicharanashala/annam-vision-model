## 12/2/2026

Looked through some research overviews initially. The main focus seems to be on hybrid models (CNN+ViT+SSMs like Mamba, etc) but aside from this, localization is usually better at detection than classification when the images are not curated specially like PlantVillage. Will continue looking into existing literature.

The major research focused on a few things:
PlantAIM which combines CNN features with ViTs attention, getting 38% on PlantDoc (seems like it is a difficult dataset?). Their approach was also similar by training on 80% train set from PV and then testing on multiple datasets.
Another is straight up fine-tuning YOLO. This also included some changes like adding SS2D (VMamba) in the neck of YOLO, which gave around 4% improvement. 
Also SAM seems to be used more recently (segmentation seems to be better than bounding boxes for this task).
I'll need to find comparable metrics with PlantDoc to see if it is just that difficult a dataset to get around. If the latest research is unable to do well, we'll have to assume that as an upper bound for now. I'll continue readi…

Using plant-specific datasets seems to be performing better but the lack of conformity and standardization in datasets is pretty common. It's difficult to truly compare them unless you run with all different kinds of data. Perhaps this can be a way of verification? Testing on different datasets that is.
Synthetic data with image generation models also seems to be done quite frequently from what I read. Approaches like including 10% synthetic data for rare classes is common.
Although not a lot, a good bit of papers do actually use PlantDoc for "real-world" verification and much less for training.
There are also highly specialized datasets for Cassava, beans, bananas, tobbaco to name a few. Depends on commercial usage?
A promising paper, Swin-HViT used a mix of 3k images from cron and 1.8k from PlantDoc to train and provided 81% accuracy on PlantDoc. Although they didn't validate on it separately, this is still very nice. Most likely approach which I will try. Will go through a few more papers to see if anything else is better than this.

Multiple other models, specifically EfficientNet-B3, ICVT, MobileViT+LeafyGAN, and a few others performed approximately ~80% accuracy but all have used a mixed dataset. I could check out any experimentation code repos if they are publicly available. They all used PlantDoc for training among other datasets so it's not a perfect comparison against PlantAIM. This could be a point of study; training PlantAIM on a training subset of PlantDoc. Assuming it hasn't been done before of course.
Object detection models can also be tested later on. A few I have seen which can get 40-60% mAP

## 13/2/2026

After looking at some more models, most of which focused on highly specialized datasets (still worth testing them later), selected the Swin-HViT to understand first. Looked into its code repo for a look-through. Continuing with a slightly deeper study of the corresponding paper

Had a progress meet. Followed this with looking through the PlantDoc paper. It seems to have also been designed for, and benchmarked on, object detection problems (contains both bounding boxes and image classes). I'll initially start with classification only and then test with YOLO or a hybrid.
The benchmarks also showed very low scores, which basically confirms that it is, indeed, a difficult dataset. Would be interesting to see the difference when it is used to train vs purely for validation

Started looking through the paper for Swin-HViT.
The main idea is that the Swin transformer is good at local understanding (kind of a like a CNN with attention) and ViT as we know is good at global understanding.
It uses something called a shifted window in the Swin part for the local understanding. 
I'll continue with a more in-depth read through of the paper

Went through the methodology and architecture of the model and how the fusion layer works. I thought it would be special but it turned out to be simple concatenation from the ViT and swin transformer. Currently going through their experiments and results

As expected (and thankfully), they used torch and huggingface Transformers, and the code is available I believe. 
Pretty interesting how AdamW has become so standard, need to study how it differs from Adam sometime. 
Used a constant lr, one experiment can be to implement a scheduler to try to improve performance
They used 15 epochs in one experiment (my sense of necessary epochs have been messed up while training diffusion models :)), while some others have used 50-60 as well. Can check while recreating.
There are also descriptive class by class metrics which I am currently going through. After finishing this, I'll create a few experiment plans and start the code setup and training

The per-class metrics are quite uneven. Quite a few have an f1 score of 1.0 while others have 0.5 or below, lowest being Tomato Leaf Bacterial Spot from PlantDoc at 0.38. While training, I need to prepare class-wise distributions as well. They will help provide info on what the model finds difficult to understand. 
Training graphs show that Validation loss converges very quickly while train keeps reducing. This dataset is quite difficult for classification if the converging val loss happens within <5 epochs.
Precision is slightly better than F1 for most values, but for this problem recall is more important and that is similar to F1, so not too good hmm.
Despite everything though, its overall results are pretty strong. We can test later by only training the model on the difficult classes to see if it learns any better.

## 14/2/2026

Looked into the paper's code in github. Had the daily progress meet. Setup the code in kaggle, but they had merged their data and uploaded and used it from their gdrive, so the code requires some updates to load the data from kaggle and merge them within there directly. Currently working on this.

There were quite a few bugs while setting up the merging code but it was done and I started the remaining script. 
Will look into the code to see how their implementation functions while it runs

Went through the initial part of the code and which exact features they used from each of the 2 transformers, as well as the exact classification head used.
As well as the exact hyperparams used.
I will continue looking into the detailed workings of the Swim transformer while this one trains

The training script stopped because of a network error so I restarted it
It completed and shows similar results to the paper's which I will analyze in greater detail.
I looked into how the Swin part of this hybrid model worked
I'll also begin planning some experiments to try to check the limits of this architecture.

Went through exact training loop for the code and understood in a bit more detail how swin transformer and ViT worked. Also planned potential experiments to run with this codebase as a baseline. The main test is to train without plantdoc and then validate on it only. That is, train on the corn data where their labels match / use Plantvillage instead for greater matching (since the labels might not match with corn/maize). Another is to test with a mix of PlantVillage and PlantDoc and then validate on PlantDoc.

Made the necessary updates in the data selection code, selected the overlapping corn classes (a total of 3) for the initial test. The data will train on the entire corn dataset (not just the 3), and will then be validated with just those 3 on PlantDoc.
Made the necessary updates to separately load PlantDoc and validate on it.
I can test how much it improves (current hypothesis) if only trained on the 3 selected.
Currently the model is training.

## 16/2/2026

Looked into the results. The previous one validated on the Corn/maize leaf dataset only since the PlantDoc validation script had some errors (thankfully it was at the end so didn't mess up the checkpoints). It obviously performs well on the data it is trained on with 0.95 acc. Had the daily progress meet. Continued with the same idea for setting up PlantDoc. I had the checkpoint saved so currently setting up loading the saved checkpoints in kaggle, will then add the validation on PlantDoc to see how it performs.

There were some problems with how the checkpoint was stored so currently still working on loading up the saved model. Will need to fix the checkpointing code later.

Fixed it and tested. And the results are 35% accuracy. Hmm. Pretty interesting how bad it turned out, barely better than earlier ViT but that was on PlantVillage and a larger dataset so for comparable results I suppose it should be trained with PlantVillage again. I'll set that up with the same overlapping classes and see the results.

Updated the earlier notebook to use the Swin-HViT class instead of ViT and started training. Currently running.

Monitored the training which can take a while longer.
Looked into synthetic data generation and since I have experience with diffusion, looked into related research. They seem to be considered better than GANs here as well with the same reasoning of mode collapse being highly likely in GANs.
Further study pointed to standard approaches including Latent diffusion, repaint, pix2pix. I'll look into these deeper later if needed.
Then started looking into the other models which performed well on PlantDoc, specifically the Efficient net variant

The model finished training and the result was... Sad. Only got 30% val accuracy when trained on PV but tested on PD. Not too far from my guesses but still pretty sad. For a more comparable comparison, I'll train it again with a mix of PV and PD and then test separately on PV and PD.
Meanwhile I looked into EfficientNet-B3

## 17/2/2026

Looked into the differences between the hybrid vs normal ViT (earlier experiment) for PV train and PD val and the difference is of 5% with a 20% increase in performance from 25% to 30% accuracy. This does indicate clear improvements but the magnitude is still low.
Class wise accuracy is much better distributed though with no high peaks but everything settled around 50% with a few low acc in some classes. This seems promising other than going for YOLO.
Had the progress meet

Setup the merging code and train-val pipeline to validate on PV and PD separately for a better comparison. 

The results are slightly better while it's currently at epoch 6. The highest accuracy for the held out PD samples is 62.75% while PV held out have 99.8. the 62% is significantly better given it has 12 classes to corn + PD's 4

The model completed training. The highest amount still remained 62, and it started to go worse for PD.
It did slightly improve for PV but that's not as useful.
Added the rice disease datasets that were shared after kaggle decided to finally show them after I spent a long time trying to add them.

Looked through the class wise dataset distribution for the given train and test sets.
It seems there is a decent bit of imbalance with classes with 80 as well as 5k samples 

Would require some work. I'll start without anything different, and then slowly attempt ways to fix the imbalance to see if anything improves
The test dataset, in comparison, is pretty nice with either 15 or 30 samples per class

Created the Dataset class to cleanly load the data and apply transformations
Split the train set into train and val loaders with 80-20 split
Partially implemented the transfer for the Swin-HViT model and train script

## 18/2/2026

Mostly completed the training script except for a few bugs.
Had the progress meet

Fixed it, setup training logs (tqdm only ofc) and started training. Could take a while

For treating the imbalance, started looking into diffusion related research for synthetic dataset generation

The training is currently partially through epoch 3/10. It takes a while to train.
But the results are already pretty good with a 90% accuracy on the validation split. The test results will be done after the training finishes ofcourse
Also given the imbalance, i'll also get some other metrics like F1 taken over all classes and per class precision and recall
I have a feeling there would be some problems with the underrepresented classes

There was an electricity cut and the session was interrupted before I could download the checkpoint :)
It's started again and the model is again at epoch 3 (mostly complete with epoch 3) and I have downloaded the checkpoint this time  :)
The other techniques for dealing with class imbalance are mostly clear and need only be implemented based on how the model performs, but in case it needs a bigger way to deal with it, data augmentation through generation from a Diffusion model would be useful.
So while the model trains, I'll be going through a new paper in Diffusion called Back to Basics which deals with some minor changes in the architecture of these models to get a better idea of the available literature

There was another bug in kaggle where it for whatever reason stopped 
Thankfully I had the checkpoint so started training again.

As for the paper, the main idea is to move away from the noise prediction done in most of the literature from 2020 to do x-prediction to model the low dimensional manifold in the full space