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