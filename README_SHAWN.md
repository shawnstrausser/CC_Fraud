Title: Kaggle Credit Card Fraud Dataset 

This repo trains a logistic regression model on credit card fraud data to predict whether a transaction is a fraud or not. In this data set, each row corresponds to a transaction and features associated with it, including the amount, a time stamp, and a set of x dense features which have been modified to maintain privacy. This dataset has a number of useful properties: it has x rows and is x MB So that on a home laptop, it requires some computing time. Furthermore, it is severely imbalanced, with only x samples (x%) which are positive. 

The key files are:
data.py Reads the credit card dataset and splits it into a train/test (80/20) that is time-based. Notably, the rate of fraud differs between the two, suggesting either a drift or small sample noise. 

Usage: python data.py <creditcard_csv> <output_dir> 
Writes train.csv and test.csv to <output_dir>.

train.py

Focal loss? 

The EDA notebook investigates the transaction rate and fraud rate throughout the dataset. It also plots log-scaled amount for positive and negative samples. 