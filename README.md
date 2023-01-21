# 3W_TFM
This is a fork of the [Petrobrass 3W dataset](README_original.md).  
# UOC Data Science Master TFM (Final Master Project)  
  First semester course 2022/2023
## Aim of the project
  Classify the undesirable events that occur in offshore oil wells reported in the 3W dataset (Petrobras) using the techniques and algorithms of machine Learning.
## Steps
### Exploratory Data Analysis
  See [EDA](tfm/EDA.ipynb)
### Preprocessing
  The dataset is composed of 8 folders, one for every situation of the process (one normal and 8 abnormal). Every folder has multiple files containing a time series of meter readings and a class label.  
    
  A data manager Python class (d3w) has been written to facilitate access to the files and split into training, validation and test datasets.  
    
  Two main assumptions has been made prior to fitting model to the data:  
  *  Minute averages instead of instantaneous (per second) data are used. It seems that the dynamics of the process are not so fast.
  *  The situations can only be assessed looking at a time window of around 15 minutes (2D time x feature).  
  
A Data Generator class (CustomDataGen) has been developed for every type of model.  
### Modeling  
#### Keras models  
  A Python class has been developed to fit, evaluate and score different neural networks. The most promising are CNN's (Conv1D layers) and RNN's (LSTM layers). See [keras_model_complex](tfm/keras_model_complex.ipynb).
#### Sklearn models
  Some models that accept incremental learning has been tried not very succesfully. See [multiclass_model](tfm/multiclass_model.ipynb).  
  A better solution has been first preprocess the dataset through a ipca model (incremental PCA) and compress the data into a much smaller datasets that fits in memory and allow the use of a greater variety of models. See [ipca_model](tfm/ipca_model.ipynb) for a Random Forest.  
#### River models  
  [River](https://riverml.xyz/) is a library to build online machine learning models.  
  
  Preprocessing is carried out using River routines:
  *  Trying to emulate a [SCADA](https://es.wikipedia.org/wiki/SCADA) that send to the model one row every second. Data is preprocessed (scaled), incremental rolling time window statistics are calculated and fed to an incremental learning model. What's not very realistic is using an abnormal class label for each row as this classification has been assessed by an expert in batch. The score of the test dataset is poor and the fitting process is very slow.  
  *  Using minute averages computed with pandas is much faster but less realistic. A python class for rolling window values has been developed that is used along with river stats calculations.  
  *  Only real data sorted by date has been used. The only abnormal situation with enough real data is Flow Instability (class 4). 
  
  An Hoeffding Tree Classifier has been used. Is a multiclass classifier.  
  
  A drift detector has been introduced in the stream learning process trying to detect concept drift in the flow of data that advices to reinitialize the model been trained. See [river_window_model](tfm/river_tfm5_4.ipynb).
 
 ### Docs
   [The final TFM report](tfm/Docs/TFM.pdf) in PDF format (Spanish).  
   
   [The final TFM slide presentation](tfm/Docs/TFM_slides.pdf) in PDF format (Spanish).     

  
***  The tfm folder contains the ypnb files for training various models with the dataset.
