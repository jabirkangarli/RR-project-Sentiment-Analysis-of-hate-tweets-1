# RR project - Sentiment Analysis of hate tweets
Reproduction of an already built RNN machine learning model. The goal of this project is to find possible flaws in the model by using a different dataset in order to improve and optimise it.

## 1. Problem statement
Twitter is one of the biggest and most popular social networks with around 300 million users, 40% of which are active on the platform multiple times per day. Hence, it’s understandable the need to monitor the sentiment of the users, especially if it can help stop the spread of hateful and damaging massages.

The goal of our project is to reproduce the original project and to replicate the results using the dataset described below, with the aim of building a robust model for predicting hate tweets. The model is based on Recurrent Neural Networks (RNNs), with the implementation and evaluation of different layers such as LSTM, GRU. Furthermore, we used pretrained word vectors from GloVe model trained with tweets, which is loaded in our model in an Embedding layer.

## 2. About the dataset
The data used was found on Kaggle:

https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset

The dataset used to build and evaluate the model contains 56744 rows and three columns:

id - ID assigned to each tweet
toxicity - 1 if the tweeter is considered hateful and 0 non-hateful
tweet - string containing tweet’s text.
Additionally, in our repository it is possible to find a diferent test dataset with 17197 unlabeled tweets which we will try to label in the end using our final and improved model (this last dataset comes from the original project).

## 3. Model development
The model was developed using Jupyter Notebook (6.1.4) and Google Colaboratory platforms. 

Packages used in the model:
 - python (3.8.0)
 - keras (2.9.0)
 - nltk (3.5)
 - numpy (1.22.4)
 - pandas (1.0.5)
 - matplotlib (3.3.2)
 - regex (2020.10.15)
 - spacy (3.2.2)
 - tensorflow (2.9.1)
 - wordcloud (1.8.1)
 - scikit-learn (0.23.1)
 - sklearn (0.0)

We used GloVe as the pre-trained word vector, which can be downloaded from the following link: 

https://nlp.stanford.edu/projects/glove/.

More specifically, we used **glove.twitter.27B.200d**.
