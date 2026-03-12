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

## 19/2/2026

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

The training seems stable around 92-93% at epoch 4. I'll let it continue till epoch 6 (and not till 10 since each one takes quite a long time). 
The back to basics paper seems to cover reasonings about previous work on manifold learning in ML. Even aside from data augmentation, this might be interesting in training the classifiers as well. The idea is that a bottlenecked model might be better than a larger model if the important data lies on a low-dim manifold. A Vision Transformer with larger patches could be good. I might give it a try in the future.

## 20/2/2026

Completed training till epoch 5 only since it seems to have converged mostly (will run for more epochs). The testing accuracy after this is 87.1%. The per-class results are actually pretty good and not affected by the large imbalance. The only problematic one is Leaf Smut with 0.18 F1 score. From the Confusion matrix, it seems to be heavily misclassified as Leaf Blast instead. Will need to look into the dataset in more detail to understand. Another thing can be the low-sample classes were overfitted
Had the progress meet

There was an increase in val accuracy at epoch 6 to 94.6% but none at epoch 7. Currently 8 is training. I will go at 10, verify test accuracy, then train till epoch 15 before stopping if there is no improvement. Meanwhile, continued reading the  Back to Basics paper for more details regarding the bottlenecked ViT

Epoch 8 improved the test accuracy to 89%. There is hope that it will increase further. leaf smut also improved significantly to an F1 of 0.6. I will now train in batches of another 3 epochs.

At epoch 11, the val accuracy is mostly stable while the test accuracy actually went down to 86% with Leaf Smut also at F1 of 0.24. This implies overfitting, but I'll completely verify by training for 2 more epochs. If it remains low, then that means the maximum is 89% test which is not entirely bad.
Meanwhile the paper mentions that the bottlenecked ViT approach is more helpful to large dimensional datasets with low dimensional actual information. That does match our datasets which are scaled to 224x224 and I assume contain the data in a low-dim manifold.
I'll continue in this direction while it trains as well as then later testing on ViT and Swin separately to compare as discussed in the meet today

Given that epoch 13 resulted in 88.3% test accuracy, I would say that this has mostly converged (Leaf smut F1 at 0.5). Started implementing just training the ViT base version directly with the same splits. I will also setup another account with the rice disease dataset after this is running and setup Swin-tiny (used in the current hybrid) separately to train as well. 
Actually nevermind, my main account's weekly training quota is full so I will train on the different account, as well as see if it is sufficiently fast locally.
Next week though I should be able to run 2 accelerator sessions
The diffusion paper went into the differences between different combinations of x,e,v-prediction vs losses and that they are all valid generators (all 9) but are not the same.

Setup training for the ViT on another account and started training. Currently at epoch 3 with 89% val accuracy, a little lower than swin-hvit and I guess I can let it run after checking out as well. Though I doubt it will perform better than the Hybrid model.
Also created a 3rd account but it needs some verification and time before I can use the accelerator, so I will fully set that up by tomorrow.
I suppose running 2-3 experiments parallelly will speed things up

## 21/2/2026

Finished training till 10 epochs. The test accuracy is 87.5% with Leaf smut f1 at 0.64.
So its just slightly worse than Swin-HViT. Hmm. I wonder how tiny performs. Setup training for Swin tiny and currently at epoch 2 with val accuracy of 90%

Swin tiny completed training and it actually showed better results than ViT base at 88%. But I see that Leaf smut F1 is horrible at 0.12. So the overall accuracy was  better but that one class was still quite bad.
Is this a problem with the width (tiny) or the architecture?
I'll run ViT-tiny as well and see how it performs

ViT tiny isn't officially uploaded by google (the other one was a community upload which we could potentially test later), but there is a close alternative in facebook's deit tiny. It's the same architecture but with a different distillation method. Set up the code and started training

deit tiny finished training at 81% accuracy and an okayish F1 for that class of 0.29. Hmm, so a smaller one didn't help. I can compare the architectures of these ViTs with the one used by the diffusion paper and see how they differ exactly to implement the bottleneck.
Also setup the 3rd account. Currently looking through the paper's codebase to see their architecture. Once there is a direction, I will start implementing and training.
Meanwhile, I can run ViT large. Currently training that.

vit large takes about as long per epoch as swin-hvit. Pretty heavy. currently at 91% val accuracy.
As for the bottlenecked implementation, it's called JiT (Just an Image Transformer). The name comes from how the Diffusion space has been building highly complex approaches like latent space autoencoders, velocity losses, and other things, when this approach is just a transformer with a bottleneck at the beginning, large patch sizes, and it does x-prediction (relevant in diffusion, not so much here).
I could try 2 things: Just adding 2 linear layers at the beginning for a bottleneck eg. 768 -> 32 -> 768 or maybe something more lenient. Or this combined with patch sizes of 32x32 instead of the current 16.
The outcome can be either very good or horrible. It is possible that the model ends up ignoring the small lesion features in the patches if they are large and are drowned out. The hope is that it does the opposite, which I believe can be possible. 
I will begin with implementing this.

Setup the bottleneck in the patch embedding layer.
To allow these new weights time to train, implemented a warm-up phase where the other pre-trained weights from imagenet are frozen and then unfrozen after 1 epoch. Took a little while to get the surgery right. Currently training. 
also the Vit-large one seems to be stuck on 91% val accuracy.

## 23/2/2026

The vit large training took quite a lot of time and the results were not any better with the val accuracy of 92% at epoch 10.
The bottleneck approach seems to not have performed a lot better (only 77% test accuracy) but that might have been due to it requiring either more epochs or more training data for the new patch embedding layer. In case it is the former, I will continue training it. I will make a few updates to increase the warm up phase to 3 epochs to see a comparison on initial epochs.
Meanwhile, I'll look back into the code and paper for that architecture to see if I missed anything and which is why this didn't perform as well potentially
Had the progress meet

The bottleneck model when warmed up till epoch 3 currently shows 78% val accuracy on epoch 6. The curve is slightly better than earlier but it needs to be verified. 
Also looked into the way they setup the 32 patch size and it should be doable to use google's existing patch 32 ViT.

There was a network error. Thankfully I was able to download the checkpoint but it took a while
Then implemented the resuming logic and uploaded the 1gb checkpoint. Pretty heavy.
Will then restart training till epoch 15

Training continued
I checked test accuracy at epoch 10 and there's a big improvement to 80% so the longer warmup definitely helped. 
Another very important thing of note is that Leaf smut F1 is not the lowest, and is infact pretty high at 0.8
Does this imply the bottleneck helped this very clearly?
But conversely, sheath rot is much lower at 0.24
Maybe we require multiple models for different subgroups of diseases that are similar.

Currently it's at epoch 13 with 12's val acc of 84%

Test accuracy at epoch 16 was 81% 
Hmm I would say it's mostly converged.
From what I can see, I need to look deeper into ViT base and compare it with the paper's JiT-B.
In that one, their results in 256x256 were pretty good even with patch size 16.
What they did note was that the benefit decreases on lower res images because the hidden size (768) is much greater than the patch din (4x4x3 in their case) 
Of course we aren't dealing with low-res (224x224 is sufficiently high-res) and so there should be benefit.
Hmmm, I will look into this since patch size 32 might not be necessary. They only used that on 512x512 images.
Now of course things might not transfer over well from diffusion (which was their goal) to classification, but like before, it's worth a shot.

So I let it run for some more epochs and the results are not improving. 
From what I can see, JiT-B's internal transformer doesn't look different from the original ViT-B.
I can try and setup their code if the pre-training is not going to be a problem

On that note, I noticed that I have been using pretrained models, and this does cause some issues with domain shift. Would training from scratch on a smaller model help?
Either way, my understanding from today's work is that there is some potential in the bottleneck approach and I can try training a smaller model from scratch

## 24/2/2026

Setup training for ViT tiny from scratch. Depending on how this trains, I'll have an idea of if from scratch is feasible or not.
Meanwhile, I noticed the paper has some more detailed experiment logs at the end (should have noticed before I started going through their repo) so started reading through those.
Had the progress meet

The model is steadily training with aval accuracy of 73% at epoch 7.
It is looking decently promising as well actually, though nothing can be said

Also mostly set up the bottleneck patch embedding layer to add in this and will start training when this happens.
While these 2 train to like 15-20 epochs, I will look through the paper's experiments

Setup bottleneck on tiny and started training from scratch.
As for the normal tiny, it is still increasing with val accuracy of 74% on epoch 10 and test accuracy of 67.5%. I'll train it till epoch 20 to observe changes. Meanwhile the bottleneck one will also train.

The normal tiny has test accuracy of 75% at epoch 20 and horrible F1 for leaf smut again. I don't think this will get better so I will leave it be
In contrast, test accuracy for the bottleneck tiny is 70% at epoch 10, but that can also imply that it is just taking a while to understand which features to keep and extract through the bottleneck.
This is strengthened by the fact that the val accuracy is increasing in concert with training at every step.
Also from what I researched right now in manifold learning related articles, this seems to be a likely possibility in many training scenarios.
Hence I am letting it run to epoch 20, and potentially even 25 or 30 if it shows results.

Another thing of note is that it has extremely well balanced F1 scores with only a few bad recall scores.
Much better than the non bottleneck tiny model.
This is the biggest feedback for the benefit of utilizing this in the patch embedding.
It's not possible that other models haven't implemented this obviously, so I can try looking into other vision models that implement some form of bottleneck (aside from VAEs ofcourse)

Yep, there's a clear improvement in the bottlenecked one. I am quite surprised after a somewhat success after such a long time (still not better in magnitude).
The test accuracy after epoch 20 is 79%, with a clean F1 at that (0.5 being the lowest for 3-4 classes). But it does match all of the theory I read during this time.
Also the val accuracy has still been improving during this time, proving things further. Although the progress has slowed, but it is still there.
Thus, I will continue training until epoch 30 minimum to observe the results.
This has clear potential. Funny how an old idea was observed while studying a new paper on a very different topic.

The model is still improving steadily with a val accuracy of 84% at epoch 27 (test will be checked after 30)
I have some other ideas to explore here parallely, mostly with SSMs (Mamba2 and related hybrid models with cnns and vits) and looked through some possible architectures
I'll continue studying them in more detail and then implement and train.

## 25/2/2026

The tiny bottleneck model got 77% accuracy at epoch 30. This is not ideal ofcourse because it might imply overfitting, but the val accuracy is still increasing slowly, so I am going to let it run till epoch 40.
meanwhile, I'll run another experiment with the bottleneck dim set to 32 instead of 64. Letting this run till epoch 20 will give a much better idea of how things are going. 
Also, I'll look into State Space Models in general first, followed by specific implementations in Plant Disease Detection. If bottleneck vit doesn't help, I'll try this.
And if this doesn't work either, its time to move on to Object Detection / Segmentation.
Had the progress meet.

There was an electricity near the end (it's really weird for this amount of cuts happening honestly not expected) and no checkpoints were saved so I'll need to redo from epoch 30
But training for tiny 64 bottleneck improved significantly to 88% val accuracy and 32 is also training well 
It's worth running 64 till 40-45 as well to see improvements (after it comes back :))

I guess the random seed is not working correctly because the val accuracy is different but 87.8% is not far from 88.4 that was previously at epoch 31. Currently training, and I'll download more checkpoints in case there's another cut.
Also, the 32 one is at epoch 7 with val accuracy of 71.4%, consistently increasing with train accuracy. I have high hopes for that as well when it reaches epoch 20. Meanwhile looked into the idea behind SSMs

The 32 bottleneck one is performing similar to 64 with test acc of 70% at epoch 12 with similar F1 per class scores
Training is going smoothly 81% val accuracy at epoch 18.
As for the 64 one, it has a result of 79% at epoch 45
I'll let it run a while longer and see if it improves because it might be converging.

The 32 one improved at epoch 20 with test accuracy of 79%, very similar to 64. very good.
The earlier 64 one seems to be stationary around 88% val accuracy only, currently training epoch 49. thankfully train isn't increasing either so it might not be overfitting yet
Looking into other models

Hey the 64 one actually gave test accuracy of 80.6% on epoch 50. Not intense, but an interesting thing. Let's see how much it gives at 60
Meanwhile the 32 one is consistently improving with 83% val accuracy on epoch 27
Hybrid models with SSMs have the benefit of being quite fast from what I can see. A potentially good approach to look into

Remembered that I hadn't gone through the back to basics paper's appendix part yet. Did that. Nothing too useful, most of the stuff is for the diffusion model part rather than the manifold learning.
The 32 one remained at 79% at epoch 30 like 64. Seems to be going through a similar curve. 
Hmm I wonder what that implies. Is there potential to go lower? Or will this one perform better with more epochs? I will test it out

Also hybrid approaches with one part being an SSM seem like a potentially good experiment 
I'll set things up tomorrow while the other 2 continue

## 26/2/2026

The 64 one showed an improving test accuracy of 82% at epoch 60 
I wonder though, if it might have overfitted and the images are not diverse enough. Hmm, Wonder if PlantDoc has the corresponding rice disease images. Could also try to find some of my own
The 32 one was also promising. 
To verify how a smaller bottleneck performs, I will setup experiments for 8 and 16 sizes as well.
Started looking through Mamba as an architecture
Had the progress meet

SSMs were initially designed for sequential tasks, later appropriated for language modeling, and then much later used for vision tasks with 2d versions like VMamba
Their 1d version (started from the beginning for a better foundation) are very similar to an RNN but with one major difference.
The way their hidden state is computed makes it easier to parallelize, fixing the major problem with RNNs. 
In fact, you could say they are similar to transformers more than RNNs because of how they parallelize.
They are also discretized from a continuous representation in comparison to RNNs which are inherently discrete 
It also uses some initialisation strategies on its A matrix so it can store memory much better

The 8 and 16 models are also running well currently at epoch 9 with val accuracy of 73 and 72% respectively 
They will be run on test data after 20 epochs

Bottleneck size 8 performed worse with a test accuracy of 73% compared to 16 with 76%. Similarly for the class wise F1 scores.
But the difference is not too much and they are still improving
I do notice that they are not performing as well as size 32, though this can also be because of taking a while to understand how to correctly compress

I'll run till epoch 30 and then decide further after that

Gained a decent high level understanding of SSMs for sequential data. Now going to start with vision SSMs.

Both bottleneck models are currently training, will evaluate with test set on epoch 30

SSMs for image classification all approach in a similar approach.
Like ViTs they create patches, and their 1d approach of sequential checking is converted to using multiple such linear scans. Like left to right, up to down, for each pixel and patch.
Multiple such models in use but I will get into more detail for Plant disease detection used in papers
I remember VMamba was used in one.
Other than that, all the theory was pretty interesting to read through 
Their main benefit seems to be understanding global context but in linear complexity instead of ViT's quadratic.

The other bottlenecked models are almost to epoch 40 and will then once again be tested on the test data. Should be interesting to see
If nothing helps, guess we need to reconsider stuff.

Researched and found a few promising papers which I will begin with tomorrow
The models are at epoch 38 and will complete in a while. After that, I'll have a better idea of how to move further with these ones

## 27/2/2026

Partially understood the architecture of VMamba

Epoch 40 finished and the result is not bad. Bottleneck size 16 showed 81% test accuracy, the highest at that epoch.
It's worth running it for another 20.

Had the progress meet

There were quite a few problems setting up the checkpoint with my pc crashing in the middle as well.
Fixed them and training is continuing

Training is at epoch 42 with val accuracy of 85%. Decent.

VMamba is actually pretty recent from late 2024. 
The architecture is actually quite complex. Might take a while to fully understand. And I should look into existing and pretrained versions.
It should also be faster to train than vit

It was pretty confusing at first but the idea is decent.
The image is separated into patches, and these patches are unwinded in 4 different ways from top left to bottom right and bottom left to top right (all 4 combinations)
Each unwind set of patches is sent to the s6 layer where it learns the relations between the tokens 
These are then combined into 1 spatial patch which is again sent further.
In between linear projections also increase the number of "channels" available similar to how a cnn does it. But slightly different.
I'll look into a bit more detail before starting code.

Bottleneck 16 is at epoch 56 with 86% val accuracy. 
Hmm, not as much of an improvement as I thought but we will see the real results when it evaluates on the test set

I think I have a pretty decent understanding of VMamba at the very least now.
Will start by looking into pre-trained models and fine-tuned and then moving to training from scratch similar to current experiments

It finished training and the test accuracy after epoch 60 is 79%
Damn. And the F1 scores got worse too.
Hmm, i suppose 82% is the limit for ViT tiny here on this data.
And the one at epoch 40 was pretty well balanced as well.
So if we need to test things out, the epoch 40 one is alright.
Of course the best I found is still Swin-HViT at 89%, though I forget how it's per-class F1 was. If it was not equally spread, then this one might just be better.

I looked through the authors' code, and got an idea of how they are doing things (surface level mostly, pretty complex setup).
Then I researched and found mamba_ssm. It's a library with the major mamba related code ready using Pytorch itself. 
Looked through their example code and created an outline for how to implement VMamba with this.


## 28/2/2026

I had some doubts regarding why there was a need for a special cuda kernel, so I looked into that. Pretty interesting.
Had the progress meet

Setup training and started.
The necessary cuda kernel compilation is expected to take 20-30 minutes. 
During this time, I'll read into existing research further

Damn. This library is not at all structured or maintained well for working on kaggle notebooks atleast 
I have been trying to have this run but there's always another bug when the previous is fixed.
I'll try it again after the meal break but if it still doesn't work I'll have to look elsewhere.
And if it is still not alright, then I'll have to move to another model approach

Yeah no this isn't going to work.
I'll need to go back to the drawing board. What else can I test. Or should I move onto another model.
Maybe look into other people implementing VMamba on kaggle specifically?

After a long while of searching, I found that the problem might be with using the P100 gpu for acceleration in kaggle. Why?? 
Anyways, I have set it to T4x2, and also made the installation commands slightly better.
Now what remains is to let it run and see.
Meanwhile I'll look for alternative experiments if this doesn't turn out well

The updates are making it run at the very least but I don't think it is still compiled correctly.
This is because the small VMamba is taking as long as ViT-large per epoch. Not ideal.
Created the Monthly report.
VMamba is currently almost done with epoch 2. Epoch 1's val accuracy was 55.7%. Not bad. let's see if it performs okay, and then we can see how to fix the cuda compiler so it can show its major strength in speed.
Otherwise, there is a need to move onto other models or different datasets for the previously tested models

## 2/3/2026

The VMamba model got around 75% test accuracy at epoch 10.
It was also mostly converged. 
But the problem is that it is supposed to be much faster than this because of the custom cuda compilation. 
I'll do a deeper research into why this isn't working, maybe even set it up locally to see the difference if necessary.
Otherwise, we'll move to checking the earlier models on different datasets.
Had a weekly team-wide meet with Sudarshan sir.

Yes VMamba is infamous for being annoying to work with, especially on kaggle.
But I did find a few more models, including Vim, MambaVision by NVIDIA, that are maybe, possibly, potentially able to be setup in kaggle.
From what I understand, running locally will be infeasible given my GTX 1660 has only 6 gb vram compared to kaggle t4x2's 16gb.
Let me see what I can understand.
Had the progress meet.

From what I can see, the most straight forward model will be MambaVision by NVIDIA. There should also be some pretrained weights available on huggingface?
I'll begin setting this up and see how well it works. Atleast there's still some hope to test SSMs.

Looked to existing codes and setup MambaVision
But this also requires mamba-ssm hmm
I'll allow this to compile one more time
And if this is still being annoying, I'll move onto training the previous models on different datasets

Finally tried everything and it seems no other Vision SSM works without the mamba_ssm library. 
The environment of kaggle is fundamentally unable to work with this, and locally is infeasible.
So, I'll stop further time on this.
I'll begin using different datasets on existing ones 
Let's start with the train on PV and zero shot test on PD with bottleneck size 16 DeiT tiny from scratch. Can compare with SwinHViT

Setup the training (merging the older style of my work with the better newer code took a little while), and started training 

At epoch 9, it is 81% on PV but stagnating around 8% for PD
I'll see if anything improves at epoch 20 
Otherwise, I'll move to the mixed PV PD dataset

## 3/3/2026

The val accuracy (on PD) at epoch 10 is also 8% 
And the per class accuracy is also very bad. Most images were being classified incorrectly as the same class. So it didn't understand a lot. 
But given the model is not pretrained and only at epoch 10, I have setup to run this till epoch 30 and will see then.
Parallely, I'll set up 2 other bottleneck size models on my other 2 accounts. 32 and 64 should be good.
Had the progress meet

Setup training for 32 and 64 bottleneck sizes on PV train and PD val.
The 16 one has got to 11% val accuracy at epoch 26
The 64 one is at 10.5% at epoch 16. This one is performing better than bottleneck 16 which had 7% at epoch 16. Maybe too much reduction is bad here? Will need to think more about this.
As for the 32 one, I noticed I had forgotten to turn on the accelerator so it didn't get too far and I restarted it.
All 3 will continue training while I try to analyze these results

The 16 one maxed out at 11.4% at epoch 30 with horrible F1 scores (eve 0 F1 for a few classes). It's just barely better than a random guess and seems to heavily predict all images as pepper bell bacterial spot and tomato late flight. Interesting.
The 64 one is at epoch 28 with 11% val accuracy as well. Hmm not improving much even though it got fast at the start. Maybe this is the limit and the smaller bottleneck of 16 found the same representation from more training but can't find much more.
The 32 one is at 9.5% at epoch 9. Around the same as the others. Hmm. I'll let these complete and then see further

The 64 one completed with 11% val accuracy. F1 for this is slightly better but still really bad.
The 32 one only got to 10.5%. Much worse than the other 2. But it's F1 scores are much better than the other 2. Again, still really bad, but slightly better.
So it seems zero shot on Plant Doc is way too difficult for these guys. Next step will be to make a combined dataset with PV and PD and then see how much that improves.
If I remember correctly, then SwinHViT improved from 30% to 60% with this. Let's see how much these go.
Started setting up the code for this.

Setup took some time since the scripts for the merging had been done only once for swin hvit and I had apparently not saved the notebook 
So after creating again, slightly better designed, I started training with all 3 accounts on all 3 types of bottleneck sizes 
Currently training
The validation is initially done on a mix of PV and PD.
After these train for 20-30epochs, I will test them on PD only separately 
80% val accuracy at epoch 10 for bottleneck 16, same for bottleneck 32, as well as for 64.
All are similar for now

While it trains, separated the PD part from the val set to see how it performs on that.
After training, this is now at:
For bottleneck 16 - 20.3%
For 32 - 13.7% (damn)
For 64 - 23.5% (highest, but not by much)

This makes it very very clear that Swin-HViT fine-tuned was much better than this since it got 60%+ with this setup (albeit the exact dataset was slightly different but a difference of 40% is a lot).
So this is again not a good side.
This concludes for classification at the very least, Swin-HViT is the best from the models I have tested, as was also my hypothesis at the beginning.

Now I need to look into either different datasets (plant wild?) or see some other approach to modeling.

After looking through other types of models and removing the options Vishnukant and Anuraag have tried, I found 2 very interesting types: MLP-mixer and FocalNet.
From what I understand, they are quite unique in their architectures with the commonality of trying to provide alternatives to Transformers and attention. 
Can be interesting to look into. I'll continue further study into these next time.

## 5/3/2026

Starting with MLP mixer and gained an overview of its working.
Really interesting how everything now uses patches in some form. After the patches are created, they are projected to a hidden size C (single dimensional), making the input SxC where S is the number of patches.
The architecture doesn't change its hidden size, maintaining C at all layers.
It contains 2 types of layers, one mixes the spatial locations, while the other mixes the feature channels.
This mixing aims to provide similar results as local and global understanding but with a simple architecture, while also maintaining linear complexity.
I'll now look into some major papers for the architecture and then to plant disease specific if they exist.
Had the progress meet

Understood the main idea behind the different layers, how they work, and their problems. 
For example, MLP mixer has a tendency to find sharp local minimas, causing overfitting.
This can be fixed with Sharpness aware minimiser (SAM) which penalizes such sharp drops and tries to create a flat minima.
Also, the mixer might be not as accurate as ViTs, but it has been shown to be at least 3 times as fast during inference, showcasing it's benefit. 
I will now move onto agriculture specific work, followed by some major papers.

After a bit more research, I can see that there hasn't been too much work specific to using MLP mixer on plant diseases, but there is one major paper from early 2025 benchmarking major results and their methodology.
From the overview, it looks promising, and I'll continue with it and see how to begin experimentation after it.

The paper starts with information about the various types of diseases and how they show them, and more specifically why diseases visible on leaves are of value.
This is because the others are usually physically occluded, and unable to seen, while leaf diseases provide a bit of time before irreversible damage is done to the plant.
They then move onto why specifically MLP mixer (gMLP that they also used; very similar model) are useful here: their benefit for edge ai with their compact sizes and simplicity (relatively speaking)
They then go into the architecture for the models. They had a bit more detail so I went through it. Very interesting stuff. gMLP (g is for Gated) was new so I tried to understand in a little more detail.
MLPs can easily overfit, so they implemented Gradient Centralization to combat it. I am currently reading through this, but the core idea to fix overfitting in MLPs is the same as SAM that I previously mentioned.
I wonder what exactly it is. I'll continue reading this

So, Gradient Centralization (GC) literally zero centers the gradients calculated during backpropagation. This makes things much more constrained and stable.
It also notes that it inherently converts the gradients to a slower dimensional manifold.
It stops any one feature from being dominant and causing overfitting, forcing the model to learn diverse features.
It has been empirically tested to effectively smooth the space.
How does it compare to SAM? Maybe they are similar in some way? Hmm.
They also trained on potato and wheat leaf datasets (I'll see which one exactly) 
They also trained from scratch instead of using pre trained weights. 
They saw 98% on potato and 91% on wheat diseases, with mixer and gMLP both giving similar results
All classes were pretty equal with wheat brown rust being slightly worse at 0.87.
On these datasets, these models perform comparably to CNNs and ViTs.
Sounds promising, I'll now begin looking for their code, see if existing implementations exist, and then begin training. I'll start with rice disease

After a lot of searching around, I couldn't find any open code specific to the paper.
But given that they trained from scratch, the major problem would at most be with implementing GC or SAM (I can also compare which one performs better)
So after some more searching I found an implementation for MLP mixer in timm. This should be good given its in timm.
I looked at the exact architecture in a bit more detail and it should be good. I'll begin with the code and start training next time.
Added results.md for previous experiment under its folder

## 6/3/2026

Setup training for MLP mixer small 16x16 patches.
Currently training for 30 epochs on rice disease datasets
I'll compare both accuracy and training time.
Currently it's just slightly faster than deit tiny at 10-11 minutes per epoch.
Had the progress meet.

It's training steadily and at epoch 7, it has 72% accuracy. Taking around 11 minutes per epochs, sometimes 12.
I have also found a simple way to implement SAM and GC and will try set them up to run in parallel to this run.
This will help in getting a comparison to how much they affect this model on this specific dataset.
Should be interesting

Setup SAM and AdamP for regularization of MLP mixer in separate kaggle accounts to run parallely.
I replaced GC with AdamP because a custom implementation was the only way and that messed up training times because of python being slower than the gpus.
AdamP works on a very similar method. Not exactly the same, but very similar.
Used Pytorch-optimizer for pre-existing SAM and AdamP implementation
SAM is much slower to train (2.5x) than the base approach. This is expected since it effectively does twice the work while optimizing the weights.
AdamP also takes just as long now that I look at it again.
I wonder why that is? Will try to understand while all 3 run

At epoch 21, val accuracy seems to have been converged around 72% for the base MLP mixer. I'll let it continue till 30 and then see the class wise distribution as well.
The other 2 are following the base one for now with 63% val accuracy at epoch 3 for now.
I have hope that this will increase. Now that everything is stable, I will continue looking into the exact details of SAM and AdamP and see how they differ in their approach

Yeah the main idea is still that SAM looks into the future and changes where it goes while AdamP modifies the gradient directly.
If AdamP was implemented at a lower level alongside the c++, it would be much faster for this reason. But given its in Python, it ends up becoming equal to SAM.
Currently here is the training progress:
The base model remained stuck at 72% val accuracy, currently at epoch 29.
AdamP at 68% at epoch 4, same as the base one.
SAM at 67% at epoch 4.
These ones will take a while

The baseline model gave test accuracy of 65.5%. Around what the validation got and slightly lower.
The f1's are mostly balanced which is nice, but the overall results is lower than ideal.
Now the good news is that the SAM and AdamP models have already improved past 72% (74 and 73 respectively) at epoch 8. 
This basically confirms that fact about the normal MLP getting stuck in a local minima while these 2 have moved past.
The increase in training is worth it. Especially so since the inference time for all 3 will remain the same in the end.
For now, I looked into the imbalanced classes and for this model specifically there weren't any problems caused by class imbalance.
Of course, I can also start training with imbalance fixing techniques later on if the results stay promising.
Will continue reading the remaining part of the research paper and see if I can see something new.

They are training quite slowly, with epoch 10 showing 75% for SAM and 73.5 for AdamP.
It's not been longer than 2 epochs but it might be stagnating? I hope not :D
I'll let it run and hopefully it improves. 
The paper doesn't have anything else of use that I can see. There are a few other papers, specifically one for FocalNet (the other model I'd like to try) which I will begin reading.

## 7/3/2026

The sam one showed progress by moving to 79% by epoch 17. It proves it moved past a local minima. Meanwhile, AdamP was being too slow and did not budge from 73%. So it might have also become overfitted. So, I will continue with sam for this and see where else it moves by epoch 30. I'll also see if there are any pre-trained weights available. Though I didn't find any last time with `use_pretrained = True`
Had the progress meet

At epoch 19, it seems to be inching forward to now 79.5%.
As for focalnet, I got an initial understanding, and found the main paper: https://arxiv.org/pdf/2203.11926
I'll read through this while the mlp mixer sam trains

Hmm, it is still stuck at around 79 at epoch 21. Is this the limit? I hope not.
Focalnet stands for Focal Modulation Networks. They were introduced as an alternative to self attention inside Vision Transformers. Though as can be seen from the fact that they came in 2022, they have not gained general popularity. It mainly performs 3 steps: Hierarchical Contextualization (for extracting features), Gated Aggregation (calculates what weight to give each context feature), Element-wise modulation (fused back into the original query).
I will continue in greater detail.

There was a small jump at epoch 22, but it has remained stable. The train accuracy continues to inch forward while val remains stagnant. I think it might have started overfitting, so I'll continue till maybe epoch 26 and then check the test set results.

Focalnets have a pretty good history. They performs just as well, sometimes reaching SOTA performance faster and easier, they are more efficient in terms of FLOPs, and they are more interpretable in that you can visualise their modulation maps and see how it separates the foreground, etc.
It's also very good at cross domain problems similar to CNNs, and has recently been applied to speech coding, medical images, and climate modeling.
But this raises the question of why it isn't more popular and I only just heard about this when it is more than 3 years old now.
The major issue seems to be how it needs to be more deeply optimised for it to actually achieve the high FLOPs amount.
And to put salt on the wound, just around when this was released, Flash attention was also released, and it improved the speed for self attention, effectively making one of the main theoretical points of FocalNet useless.
And so they never got too much use.
But nevertheless, I will try them out (Unless they have the same problems as Mamba, then nevermind). I will begin looking for papers focusing on focalnets used for plant disease classification and then decide how to move forward with this.

I stopped it and it did give a pretty good 80% test accuracy.
Also the confusion matrix was very clean, much much cleaner than previous methods. It was almost not affected by the imbalance.
This might be the effect of the SAM regularizer? Possible.
The lowest F1 was around 0.6 and highest at 1.00
This would be a very good model to run. Of course I will compare the per class performance with swin-hvit later.

Also found a very recent (march 2026) paper for FocalNet on crop disease called Focal-HAIN. I will begin reading this

Oh turns out F-HAIN is an object detection model. Hmm, I'll still read this. I will first test things out on a normal FocalNet classifier and then pivot to this for object detection. That might require a different dataset probably.
Anyways, the architecture is pretty neat. It is based on YOLOv5, but uses custom layers/modules: F-SPPELAN (the focal part), HAIN(Hierarchal Adaptive Interaction Network), which integrates the features directly in the aggregation path (need to see what exactly this means).
It is supposed to be be both High in speed while also being accurate.
I'll compare how the speed is, and then maybe also look into plant disease prediction results with vanilla YOLO models to see if there's an improvement. There should be a table in the papers ideally.

After looking around, I found that timm also contains an implementation of focalnet (this is pretty nice library, and I think I just read some related news mentioning the creator of this). I will begin training with just a normal classifier (not f-hain). But before this, I'll get a bit more detail about f-hain to make it easy to continue after the classifier begins training or finishes to compare results.
They mentioned an IP02 dataset. I'll look into this, probably one for object detection with diseases and pests.
And they claim to be outperforming multiple models, including YOLOv11, and claim it can be deployed on Raspberry pi 4b. Pretty neat indeed.


## 9/3/2026

Starting setting up a simple FocalNet directly. I'll first start training one from scratch and then see if there is one with pre trained weights and start that as well. 
Had the progress meet

Took a little while and started training for the from scratch one.
Also setup pretrained weights for focal net. Currently the version I am using exactly is focalnet_tiny_srf. A small model is only feasible for training from scratch.
Currently the from scratch model is at epoch 4 with val accuracy 63% and steadily improving.
The pretrained one is also at epoch 4 with 91% val accuracy. 
I guess this is the benefit of being pretrained before. Hopefully this can keep improving and doesn't stagnate like previous fine tuning attempts.
If this shows promise I can try to fine tune a larger version as well.
Also as Deepthi mam suggested, I can take a subset of 5 classes from rice disease and train Swin-HViT normally and with SAM to see if that improves the results. And if it does we can look into trying to train with all the classes.
Well for now let's focus on FocalNet.

The from scratch one is improving, currently at epoch 7 with 68% val accuracy.
The pretrained model is also slowly improving, currently at 92% val accuracy at epoch 7.

Started looking into other approaches than just leaf based, and a few other ideas are: roots and soil, sap analysis, olfactory sensors, sound based, x-rays and mri data.
Most of these require special equipment or extra work. Much more difficult than just taking a photograph of the leaves from a phone. There should be other vision based approaches as well, but the previous ones listed are still okay as potential options

At epoch 17:
From scratch model at 77% val accuracy
Pretrained model at 93% 
The scratch model is being very consistent actually 
Better than Deit tiny even if I remember correctly.
The pretrained model is still slightly improving so I will let it continue onwards as well.
FocalNet is being quite promising.
As for other techniques and problems, vision based possibilities include hyperspectral images (hm, depending on the use case of the problems this can be feasible), there's also stem based detection for a few diseases which are visible there. Could be a possible direction to test.
Root disease modeling is also possible but this will be manual given a photograph of the roots will require someone to manually pull it out.
Also models for fruits and their seeds after a harvest with hyperspectral images.
Also sfm (structure from motion) to create 3d representations of the plant part to get a much deeper understanding

At epoch 23:
From scratch continues to increase to 80% val accuracy, while the pretrained one remains stagnant at 93% peak. Train accuracy is also stagnant at 96%, so it's unlikely to be overfitting, although I don't know if there's any hope.

Other than the previous approaches mentioned, I also looked into what the current trend of the times is regarding plant health, primarily focused on computer vision. Trend is moving towards VLMs trained on image caption pairs (AgriGPT, AgriVLM, KBNet, etc). This will be a very different approach to the current models I am working on, and I think maybe this has been touched upon before I joined?
Hmm
Other than this, I noticed multiple works on 3d reconstruction (related to sfm I mentioned previously) (Wheat3DGS, Gaussian plant)

The training results are as follows:
From scratch: Final test accuracy of 72% and not too good per class F1. Nothing special. In effect worse than MLP mixer sam.
Pretrained: Final test accuracy of 88%. Actually pretty good, second only to Swin-HViT. The per class F1 is also only problematic for leaf smut with 0.23, everything else is above 0.7, and most are above 0.9
Pretty good.

After looking through the classes that were being most difficult over my previous experiments, here are the ones that I have chosen: Sheath rot (usually low F1), leaf smut, leaf blast (commonly misclassified so the pair is needed), ragged stunt virus, brown spot.
I'll begin with training swin-hvit on these with and without SAM for regularization.
This serves 2 experiments: 1. Testing the model on a smaller difficulty, 2. Testing effectiveness of sam on swin hvit for later trying on larger datasets where it can take really long.

## 10/3/2026

Partially implemented the notebooks for the Swin-HViT 5 classes with and without SAM experiment.
Had the progress meet

Took a while to update the Swin-HViT code style to the new approach and match it alongside the rice disease data, but the base training is setup and running. Currently creating SAM for training it another account

The sam regularized swin-hvit code completed and both are now training. Given what I learnt previously and now, MLP-mixer has the highest chance of improvement, followed by transformers with sam. CNNs in comparison won't have much of a benefit since they automatically are scale equivariant and invariant. I will begin looking into what else I can experiment on.

Both models are being trained with pretrained weights since this has shown to be better.
for base swin-hvit: After 15 epochs, the test accuracy on the 5 classes is 95%. Very nice. But since the val accuracy was still increasing I'll train it till epoch 30.
for sam regularized: At epoch 6, it's at around 88%.

The base model also approached 87.8 at epoch 6, so the regularized one is currently following this same path.
But from what we learnt with mlp mixer experiment, this should ideally improve along with the epochs.

Aside from this, I am continuing to look through available datasets. And very interesting how almost everything is focused on leaves. Well, it makes sense given that leaves are the most visible part and multiple diseases show their symptoms on the leaves quite clearly. Targets like stems or roots are more difficult to setup. I'll continue looking.

The base model at epoch 24 is still only at 88.8% peak accuracy and has not improved further. But the train accuracy has also not moved. This implies it is stuck somewhere but has not begun overfitting. Which is good.
The sam model is at epoch 12 and the peak val accuracy is at 88.6% which happened at epoch 8 earlier. I will run on test set when epoch 15 is completed and then see if it is worth training like 5 more epochs potentially.

As for potentially useful datasets, 3 that I have found (and still searching) are cinnamon plant stem and branch disease dataset which is clearly different from leaves and could be interesting to test out on, and fruit disease dataset 4 diseases for apples. After moving on from leaves, the quantity and standardization of datasets is quite low. Even these 3 took a long while to find. I have some ideas for finding a few more while the swin-hvit trains.

Alright fun observations.
After 30 epochs, the base swin-hvit regressed to 82.5% test accuracy. This implies overfitting. Yes, the confusion matrix shows an old problem: Leaf smut misclassified as leaf blast. Wait, does this imply that this specific problem between the 2 classes happens because these are extremely easy to overfit or confuse (or well there might be a better word). So one solution can be to separate these 2 entirely and then train the model. And then train another model specifically for these 2. They are very "sensitive" if you will.
As for the sam one, after 15 epochs, it shows a very disappointing 72.5% test accuracy. But funnily enough the confusion matrix shows that the only difference this had from the base model is that it misclassified leaf smut - leaf blast much more heavily (from 16 to 25 here). Pretty sure this alone degrades the performance from 82% to 72%.
So this basically tells me that sam here will not have any benefit really. Possibly because of the swin part in the model, which is local as well.
And that one experiment to test will be to see how things improve if I just only remove the 2 problematic classes and train the swin-hvit on remaining 17 classes.
Hmm, something to discuss with Deepthi mam
Other than that, I wasn't able to find many other useful datasets. A few papers mentioned some datasets but I was unable to track them down, so I will have to remain on the few I found earlier.


## 11/3/2026

Setup another notebook for training the 17 classes except leaf smut and leaf blast on pretrained base swin-hvit. Not sam since it didn't show any benefit last time (most likely because of the swin part which handles the same effect in a way), and started training it.
Also added the previous notebooks for swin-hvit base and sam and working on the results.md file
Had the progress meet

Created results.md for the 17 classes experiment
Setup mlp mixer after a discussion with deepthi mam for comparative training on removing leaf smut and leaf blast. 
One thing to note is that mlp mixer doesn't have pretrained weights.
Now both this and swin-hvit will be running on the 17 classes parallely
The swin-hvit is currently at epoch 4 with ~95% val accuracy.
I'll look through the cinnamon and fruit dataset to gain a better understanding of both.

The swin-hvit is training at epoch 6 with 5's val accuracy as 96.3%. Pretty good and it's improving a good amount. I should probably also show f1 scores for each epoch? But an average of the f1's might not hold too much value? I'll have to look into this.
The browser running the account with mlp mixer crashed after my pc froze for a few minutes, so that will need to be restarted, but it was steadily improving and will continue.
The cinnamon dataset is very small with only 2 classes and very few images per class. The images themselves are overall alright. This can be tested, but even the smallest of the models I am using will overfit on this. And there won't be much left to test.

The fruits dataset is a lot bigger than the size of the cinnamon one with 4 fruits, total of 17 classes, and 100-200 images per class. This is pretty good and it might potentially be enough for fine-tuning. This also means we can't run mlp mixer because of a lack of pretrained weights in timm.
I'll test it out regardless if it can actually be pretty good.
After going through the individual images themselves, I can say that at the very least, the dataset's quality is actually pretty nice. There are varied angles and the only thing common is that the fruit itself takes up the majority of the image. Which is completely fine, we can just instruct the users to take the image from a close up. Or if it is an automated system, the camera can first run an object detection and crop the fruit out.

The swin-hvit model showed test accuracy of 89.3% after epoch 10. The major issue is now narrow brown spot being misclassified as brown spot. (I will check how much the 2 classes differ). Also the peak was hit at epoch 5, so I will check by loading those weights and then run on test data.
The mlp mixer model is at epoch 5, with epoch 4 val accuracy at 75% and steadily improving. Last time MLP mixer converged at 80-82%. It would be interesting to see what happens as mlp mixer sam was actually the only one that didn't mind leaf smut - leaf blast when trained on all 19 classes.

Yep the checkpoint at epoch 5 is much superior, providing test accuracy of 92.84%. The highest I have gotten. This is for the 17 fruit classes.
The MLP mixer model is at 81.5% at epoch 8. Very nicely improving consistently. Literally every epoch has had an improvement. Very neat architecture. I remember there's also gMLP that was found to be marginally better in that previous plant disease paper testing on MLP mixer and gMLP. Might check it out later.

After a bit more searching around for any other datasets options in case, specifically focusing on fruits to merge with the existing dataset, a few appeared, but none were really mergeable directly.
In this case, I'll continue with training on the given 17 fruit disease classes. This should be quite interesting.
I'll once again train the 2 trusty models: swin-hvit and mlp mixer. I'll setup swin-hvit since the rice disease 17 classes has trained (at epoch 5 no less, need to keep that in mind). The MLP mixer I'll setup when this existing run for rice disease 17 classes with MLP mixer is finished. This will provide greater information for when the run is likely to achieve its peak

Setting up the new dataset took a while because of the differences in folder structure between this and rice disease, but it was completed.
Then I repurposed the already pretty good training script and started training for 6 epochs. There should be an idea about how it will perform. My assumption is that it should be able to work out the max possible accuracy slightly earlier than the rice disease one. It might also take a while if the classes are different or similar enough.
The val accuracy at epoch 2 is 97.8%
The MLP mixer at epoch 11 is at 83.4%. The improvement speed has slowed, but it is still decent (83 at epoch 10 to 83.4 at 11). I'll let it run until 2 consecutive epochs don't give an improvement.


## 12/3/2026

Resumed training for the 2 models. There were once again some minor issues with adding the checkpoints, but it worked out.
And the results were nice for one and not nice for another.
The MLP mixer sam on 17 rice disease classes converged at epoch 17 to 85.7% val accuracy, but sadly only got to 75% test accuracy. I wonder why it got so different from when on all 19 classes? It should have increase not decreased. But it is what it is.
Now, the new and shiny fruit dataset. The Swin-HViT trained for 6 epochs 97.3% val accuracy. There isn't any specially assigned test set for this so the validation set is the testing set in effect. The results were very clean. Also the confusion matrix made me notice that while apple guava and mango have approx 100-200 images per disease class, I had missed that pomegrenate had 1k. Very interesting.
But it's still alright because the per class f1 scores are clean as well. The lowest is 0.55 for Alternaria_mango but everything else is above 0.8, most being above 0.9
I will begin setting up the experiment for doing the same for MLP mixer. Although I doubt how far it can go without pre-trained weights now, it did gain my trust with the first rice disease results so it's worth a shot.
Had the progress meet

Partially implemented the MLP mixer fruit disease. There's a bug because of merging 2 sources which I'll fix.
Also Deepthi mam mentioned testing on the potato tomato disease dataset shared a while back for comparative results. I'll begin looking into that when the training starts for this.

Finally setup mlp mixer sam on fruit disease data and started training. Let's see how it goes.
Also downloaded and looked through the potato tomato dataset. Will now add it to kaggle and begin setting up the dataset class for my usage and train on swin-hvit. A good set of experiments for the day these are.

The MLP mixer fruit disease model is at epoch 21 with val accuracy of 87.2%. This is pretty nice. And its also increasing consistently
And it makes sense that a model trained from scratch will require more epochs than one that was fine-tuned. Though clearly fine-tuning is the better approach. This is also because ImageNet contains images of items we will look at daily, and most things follow similar low level geometry, making fine-tuning highly effective.
In contrast, something like gravitational lensing images follow slightly different geometries, colors, and the like, potentially making training from scratch better. It depends on the data and model in the end though. I wonder if there is a better guideline than just trial and error.

Setup the potato tomato dataset in kaggle and shared it with my other accounts. It was a bit more annoying than I thought to get this done. Kaggle doesn't like long image names and so I had to manually remove each one of these. Also a few images had forbidden characters when uploading to kaggle. These are probably forbidden in the linux distro (highly likely) that they are using. Also why it didn't affect locally in windows. Don't think it will result in too big of a difference in the results :). Thankfully there weren't too many that crossed the text limit and I didn't have to write a script. I will begin setting up the Dataset class for the, well, dataset.

The MLP mixer sam fruit disease got 89.5% val accuracy at epoch 30. Given that it was still improving until epoch 29, I will continue its training until epoch 45 for now.
Now as for the per class results, Alternaria Mango is at 0 F1. Huh, all of them were misclassified as Black rot. Maybe there's a mode collapse happening specifically for this class. Very interesting. Swin-HViT is still the highest available yet, but who knows maybe this gets higher accuracy, even if it ignores alternaria.

Also almost completed setup of the Dataset class for the Potato Tomato disease dataset that I had previously uploaded to Kaggle. I will then begin training as soon as it is completed with Swin-HViT pretrained.