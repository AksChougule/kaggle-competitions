## My notes/learning from Bengali-ai kaggle competition

**Competition Name**: Bengali.AI Handwritten Grapheme Classification

**Problem Type**: Multi-class image classification

**Technology Used**: Python 3.6, Pytorch 1.0.3, Jupyter Notebook

**Labelled Data**: 200,840 Images (80/20 split)

**Final Accuracy**: Grapheme model (95.5%), consonant model (98.2%), vowel model (98.3%)

**Description**: This dataset contains images of individual hand-written Bengali characters. Bengali characters (graphemes) are written by combining three components: a grapheme_root, vowel_diacritic, and consonant_diacritic. Your challenge is to classify the components of the grapheme in each image. There are roughly 10,000 possible graphemes, of which roughly 1,000 are represented in the training set. The test set includes some graphemes that do not exist in train but has no new grapheme components.


**General notes on workflow**

1. Understand the data directory.

2. Do exploratory data analysis on the shared data files.

3. For the efficiency of storage the data was provided in Parquet file format, where flattened images are stored in single dataframe.
Apache Parquet is a columnar storage format available to any project in the Hadoop ecosystem.

4. So I had to unflatten the images to create Image object and then pytorch tensors.

5. There are multiple input dataframes (parquet files), so you have to join them using concat.

6. First I tried to read directly from parquet file, but it is inefficient and time-consuming as the unflattening happens every single time.
So I converted the parquet to images and wrote on disk. Delete parquet objects to free-up some memory.

7. These images are grayscale images (with only one channel) but the pre-trained model expects 3-channel image. 
So either I have to convert grayscale to RGB using convert.RGB from PIL.Image or modify the pre-trained architecture to accept one channel. I went with the former.

8. Also the pre-trained models in resnet family expects 224*224 image by default but our original images have dimensions of 137*236. So I had to resize them to larger dimension.

9. Created a custom class for our dataset inheriting the Dataset class from PyTorch. This class enables us to access/get an image and the label, transform it using the input arguments and also convert it to RGB (3-channel image).

10. Then created the instance of transformed data object.

11. Created a sampler to randomly separate 20% of the labelled data as validation data and 80% as training data.

12. Created a datalaoder object for train and val using the sampler.

13. Then created the train model function which uses the pre-trained model for number of epochs we have specified using the criteria (evaluation matrix), optimizer and scheduler //explore more. This function also does the forward pass, backward pass, loss computation and evaluation matrix (accuracy) computation. Finally, it prints the loss and accuracy for each epoch on both train and val data.

14. Set the computation device to cuda.

15. Specify hyperparameters: learning rate, decay, momentum. Run the model. Iterate by changing hyper-parameters.

16. As we had to predict 3 separate categories of classes: graphmeme root, consonant diacritic and vowel diacritic I went with 3 separate models.
