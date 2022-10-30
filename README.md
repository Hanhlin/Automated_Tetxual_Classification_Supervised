# TextAnalytics_Supervised
Perform KNN modeling and clustering of textual data

## Instructions
Uses KNN and Fuzzy KNN.  Topics per cluster folder is generated to current directory.
The result of KNN and Fuzzy KNN is showed on the terminal.

## Sources
- Kmeans.java
- Main.java
- preProcess.java
- SimilarityMeasure.java

## Dependency Files
- stopwords.txt
- train data folder
- test data folder

## Requirements
- StanfordCoreNLP
- Jama
- gral-core

## Process 
First, read and preprocess training data to generate tf-idf, Kmeans model is trained on given document data.
Use this model to classify test data and generate tf-idf for test data using tf from test data and idf from train data.
Use KNN/ Fuzzy KNN to classify clusters for test data using clusters of the trained model.
