## My notes/learning from head-CT-hemorrhage

**Dataset Name**: Head CT - hemorrhage

**Problem Type**: Single-class image classification

**Technology Used**: Python 3.6, Pytorch 1.0.3, Jupyter Notebook

**Labelled Data**: 200 Images (80/20 split)

**Final Accuracy**: 97.5% on validation data

**Description**: This dataset contains 100 normal head CT slices and 100 other with hemorrhage. No distinction between kinds of hemorrhage. Labels are on a CSV file. Each slice comes from a different person. The main idea of such a small dataset is to develop ways to predict imaging findings even in a context of little data.

**General notes on workflow**

1. Understand the data directory. Do exploratory data analysis on the shared data files. It appeared that images had a significant variety in terms of dimension. Not just width and height but also the number of channels.

2. So as part of the pre-processing I had to reshape the images to same dimention and normalize them. 

3. Created a custom class for our dataset inheriting the Dataset class from PyTorch. This class enables us to access/get an image and the label, transform it using the input arguments and also convert it to RGB (3-channel image). Then created the instance of transformed data object.

4. Created a sampler to randomly separate 20% of the labelled data as validation data and 80% as training data. Created a datalaoder object for train and val using the sampler.

5. Then created the train model function which uses the pre-trained model for number of epochs we have specified using the criteria (evaluation matrix), optimizer and scheduler //explore more. This function also does the forward pass, backward pass, loss computation and evaluation matrix (accuracy) computation. Finally, it prints the loss and accuracy for each epoch on both train and val data. Specify hyperparameters: learning rate, decay, momentum. Run the model. Iterate by changing hyper-parameters.

6. Nothing particularly challenging about this dataset except the last 12 images having 4 channels which I had missed during the EDA and the pixel values were between 0 to 1 for some images (float) instead of 0 to 255. So multiplying with 255 and converting to uint worked as follows (img1 * 255).astype(np.uint8)

7. It is very interesting to see the validation accuracy gets stabilized at different level depending on complexity of the model. For the resnet18 the accuracy got stabilized at 95%, whereas with resnet152 the accuracy got stabilized at 97.5%.
