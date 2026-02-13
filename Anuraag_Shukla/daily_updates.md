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