# EX-06 Named Entity Recognition

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Our goal is to create a Bidirectional Recurrent Neural Network model based on LSTMs that will enable us to identify the specified items in the text.There are several sentences in the dataset, and each word has a tag.To train our model, we must vectorize these phrases using embedding techniques.

Recurrent neural networks that are bidirectional link two hidden layers that point in opposing directions to the same output.

## DESIGN STEPS

1. Import the necessary packages.

2. Read the dataset and fill the null values using forward fill.

3. Create a list of words and tags. Also find the number of unique words and tags in the dataset.

4. Create a dictionary for the words and their Index values. Repeat the same for the tags as well.

5. We done this by padding the sequences and also to acheive the same length of input data.

6. We build the model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

7. We compile the model to fit the train sets and validation sets.




## PROGRAM

Include your code here

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Sample Text Prediction
Include your sample text prediction here.

## RESULT
