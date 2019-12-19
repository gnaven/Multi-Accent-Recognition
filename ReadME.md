Source code for Multi-Accent-Recognition
Gazi Naven

This directory contains the code to run two models for classifying 
accents from audio files with spoken English speeches developed in PyTorch.

data/ : contains example data from the Mozilla Common Voice dataset
	Download it from here for the full dataset: https://voice.mozilla.org/en/datasets

Cluster/ : contains .csv files of the tensor from the last layer for each sample
	along with labels after implementing k-means clustering on them

*.pt : These files contains the network graphs of the model.

Data.py : Implementation of the DataLoader, which is developed on the inheritance
	of the Pytorch Dataset class
util.py: contains methods for loading the audio file along with methods for 
	extracting the spectrograms from each audio sample

LSTMmodel.py: Implementation of 4 layer Bidirectional LSTM model. 
AttentionModel.py: Implementtion of 4 layer Bidirectional LSTM model with 
		attention mechanism

lastLayer.py: Loads the trained model and runs it through the desired
	type of dataset. Saves the tensor from the last layer and runs 
	k means clustering with k=17 and saves the cluster labels alongside
	the last layer data

ClusterPlot.ipnyb: notebook to analyze data from the last layer of the model
	and the clusters formed. 

### The architecture of the Attenstion Model is as shown below:

! [Attention LSTM] (https://github.com/gnaven/Multi-Accent-Recognition/blob/master/figs/Attmodel.png)
