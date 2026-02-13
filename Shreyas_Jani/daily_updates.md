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

## 13/1/2026

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