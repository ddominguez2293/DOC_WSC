# DOC_WSC
Final Doc Model set-up for WaterSciCon
Author: Daniel Dominguez
Contact: Daniel.Dominguez@colostate.edu

This code includes 5 files so far:

R Markdown files are being run using the following configuration:
R 4.3.1
tidyverse 2.0.0
sf 1.0-14
dplyr 1.1.3

The Python codes are run in VScode using the following configuration:
Python  3.11.5
Python Environment Manager  v1.2.4 
Pylance  v2023.12.1 

Packages:
numpy 1.26.0
matplotlib 3.8.0
pandas 2.1.0
scikit-learn 1.3.0
xgboost 2.0.0
tensorflow 2.14.0

A OneDrive folder Symlink is used for this workflow to facilitate not re-running
the pre-processing in the .Rmd files. The OneDrive folder is managed
by Daniel Dominguez, reach out for a link. To reproduce this from scratch, run 
the scripts as documented below. 

To set up a symlink on a Mac you can use the following in the terminal:
```
ln -s /path/to/original /where/you/want/the/link
```

# 1 Aquasat Processing.Rmd 
This is the first script that needs to be run and works with the raw aquasat v1 data set (downloadable here: https://figshare.com/collections/AquaSat/4506140). You also need the level 1 ecoregions (downloadable here: https://www.epa.gov/eco-research/ecoregions-north-america).
Processess the raw aquasat file by calculating common remote sensing parameters included in the Gardner 2023 TSS study
Individual sites are then used to extract the geospatial ecoregion they are located in
Amount of observations at each site are calculated as an option for filtering later

# 2 Prep.Rmd
This is the second script that needs to be ran, this code further processes the code into individual factors and can be tweaked for mulitple parameters. 

Further prepares the aquasat dataset for training and testing
Data is loaded and then filtered based on similar paramters to other studies using the Aquasat dataset
Water Color based on remote sensing band values are also designated in this step, in the future this can be moved to the previous step
There is an option to use all the remote sensing parameters and bands "bands" or "features" which is a smaller subsection of those paramters based of previous studies for the predicted water quality constituent
data is then assignedd a magnitude of low or high based on if it is above or below the high outlier barrier  for DOC that is 40 mg/L where observations are not as prevelant
There is also a function that can randomly create a smaller dataset based on the magnitude but this has to be further explored.
The last bit of code pre-splits the training dataset selecting 80% of the data in both magnitudes for better training and testing as random sorting could include less or more valus in either phases leading to artifical low or high error rates, which the errors tend to be mostly drived by underpredictions in the high outlier data

# 3 XGrid.py 
This is the first standa alone code this is a set of code that can test many permutations of XGBoost models so that one doesn't have to test them individually.
It is good for getting a good idea of which parmaters to use but it should not be used solely for the last set of testing parameters as hyper-tuning is more delicate and encompasses more that what can be derived from these methods.

# XGBoost.py
Uses the sklearn and xgboost packages to prepare the data for training, altough pre-processing is not neccessary for XGboost models it is useful in that it can help overcome some of the biases in the data such as the non-gaussian distribution of the remote sensing parameters and true values of the water quality paramter as they are both heavily right-skewed. Encoding also helps to overcome integer encoding as having integers with all of the available ecoregion may skew with the amount of categories available. 
The rest of the code sets up the rest of the parameters needed to train an xgboost model, loss functions, tuning parameters, etc, then the model is trained and tested on the training split.
A simple graph is returned for the user to be able to visualize the output of the training.

# DWNN.py 
This code pre-pre-processes the data and then trains a deep-wide neural networks
The preprocess uses onehot encoding for the categorical values similar to pivot wider and assigns a binary input for the categories, Min-Max scaling processes the numerical features into a scale of -1 to 1 common in many neural networks as they like inputs in these ranges, it also centers the value of the water quality paramter 0 at the median of the input data
There are many loss functions included in this code which can be split into seperate files later on and calle into a central script for less bulky and faster code.
The Wide part of the architecture is a layer that is supposed to connect to the end of the neural netowork and only update the weights once per epoch which is supposed to help the neural network learn easier trends in the data represented in the categorical variables. There is an option to do a smaller hidden layers that I was experimenting to further help the network achieve better results 
The Deep part uses the traditional hidden layer feed-forward propogation which decreases the amount of nodes with subsequent layers, I have increased the amount of input nodes so that I can actually get it to make predictions as with less nodes in the first layers I was not getting any training in the DOC models just a single line across to reduce the loss function
Most of the rest of the code sets up the hyper-paramters for training, learning rate, early stopping, sample weigths, etc.
The network is trained and then a graph is produced after testing for the user to visualize the predictions. 

# DOC_RL.py

This script processes the data the same as DWNN, however the key distiction is the architecture of the DL model. Recently, I had a breakthrough of using an architechture with the hidden layers all being equal in each subsequent layer. This helps the model learn to generalize instead of memorize per the literature. Each hidden layer consists of 1024 nodes with 4 total layers. The other key difference is the exploration of the reinforcement penalty loss function. At the moment I am exploring using a penalty functon where the model is penalized at first by not making a prediction within the value of each y prediction. For example if a prediction of 1.5 gets added in and the model predicts over that value it would get penalized for not predicting, the same applies for the loss function, however these would be dynamic and weighted differently since a value of 1000 would have a much higher weight if it was not predicted within that magnitude. This helps the model learn to predict both the low values and not as severly under predict the high values like in most models. I have started walking in how much the model can get close to the actual value without it starting to freak out and go back to random predictions in the DOC model. I do this by including a multiplication factor in the loss function, for example I started with the model having to predict within the true value of the prediction and if it didnt it got penalized. Then I walked in that value, using a multiplicative of 0.9 then 0.8 and so on which lowered the magnitude of when the model applied the penalty. For now a multiplicative of 0.7 seems to do the best and although you can lower the loss and MAE on the training set by going lower it does not yield much benefit when making predictions on the testing set and in most cases actually increases the MAE on the testing set. 