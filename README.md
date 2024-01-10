# DOC_WSC
Final Doc Model set-up for WaterSciCon
Author: Daniel Dominguez
Contact: Daniel.Dominguez@colostate.edu

This code includes 5 files so far

Aquasat Processing.Rmd 
Processess the raw aquasat file by calculating common remote sensing parameters included in the Gardner 2023 TSS study
Individual sites are then used to extract the geospatial ecoregion they are located in
Amount of observations at each site are calculated as an option for filtering later

Prep.Rmd
Further prepares the aquasat dataset for training and testing
Data is loaded and then filtered based on similar paramters to other studies using the Aquasat dataset
Water Color based on remote sensing band values are also designated in this step, in the future this can be moved to the previous step
There is an option to use all the remote sensing parameters and bands "bands" or "features" which is a smaller subsection of those paramters based of previous studies for the predicted water quality constituent
data is then assignedd a magnitude of low or high based on if it is above or below the high outlier barrier  for DOC that is 40 mg/L where observations are not as prevelant
There is also a function that can randomly create a smaller dataset based on the magnitude but this has to be further explored.
The last bit of code pre-splits the training dataset selecting 80% of the data in both magnitudes for better training and testing as random sorting could include less or more valus in either phases leading to artifical low or high error rates, which the errors tend to be mostly drived by underpredictions in the high outlier data

XGrid.py 
is set of code that can test many permutations of XGBoost models so that one doesn't have to test them individually.
It is good for getting a good idea of which parmaters to use but it should not be used solely for the last set of testing parameters as hyper-tuning is more delicate and encompasses more that what can be derived from these methods.

XGBoost.py
Uses the sklearn and xgboost packages to prepare the data for training, altough pre-processing is not neccessary for XGboost models it is useful in that it can help overcome some of the biases in the data such as the non-gaussian distribution of the remote sensing parameters and true values of the water quality paramter as they are both heavily right-skewed. Encoding also helps to overcome integer encoding as having integers with all of the available ecoregion may skew with the amount of categories available. 
The rest of the code sets up the rest of the parameters needed to train an xgboost model, loss functions, tuning parameters, etc, then the model is trained and tested on the training split.
A simple graph is returned for the user to be able to visualize the output of the training.

DWNN.py 
This code pre-pre-processes the data and then trains a deep-wide neural networks
The preprocess uses onehot encoding for the categorical values similar to pivot wider and assigns a binary input for the categories, Min-Max scaling processes the numerical features into a scale of -1 to 1 common in many neural networks as they like inputs in these ranges, it also centers the value of the water quality paramter 0 at the median of the input data
There are many loss functions included in this code which can be split into seperate files later on and calle into a central script for less bulky and faster code.
The Wide part of the architecture is a layer that is supposed to connect to the end of the neural netowork and only update the weights once per epoch which is supposed to help the neural network learn easier trends in the data represented in the categorical variables. There is an option to do a smaller hidden layers that I was experimenting to further help the network achieve better results 
The Deep part uses the traditional hidden layer feed-forward propogation which decreases the amount of nodes with subsequent layers, I have increased the amount of input nodes so that I can actually get it to make predictions as with less nodes in the first layers I was not getting any training in the DOC models just a single line across to reduce the loss function
Most of the rest of the code sets up the hyper-paramters for training, learning rate, early stopping, sample weigths, etc.
The network is trained and then a graph is produced after testing for the user to visualize the predictions. 