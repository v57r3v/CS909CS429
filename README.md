java c
CS909/CS429 Data Mining 2025 Assignment 1: Classification
Your mission, should you choose to accept it, is to craft a classic machine learning solution for an object recognition task. Picture this: each object is a 28x28 pixel image. You'll get these images as 'flattened' 784-dimensional vectors, each tagged with a label (+1 or -1).
Data Sources:
Training Data (Xtrain): Rows of images for you to train your model.
Training Labels (Ytrain): The label of each image.
Test Data (Xtest): More rows of images for you to test your model's savvy.
The training data (with labels) and test data (without labels) are available to you at the URL: https://github.com/foxtrotmike/CS909/tree/master/2025/A1   
You can load the data with np.loadtxt.
��   Submission Guide:
Whip up a SINGLE Python Notebook containing all your code and answers.
Make sure it includes:
1.   A declaration (at the beginning of your submission)   of whether you have used any AI tools like ChatGPT for your work and outline in 2 lines the intention behind its use. You are permitted to use such tools as long as you declare them keeping Warwick’s values and academic integrity as a priority. 
2.   All prediction metrics, presented neatly.
3.   The output of every cell executed, so markers can verify your work.
4.   A summary table of performance metrics to spotlight the star model.
5.   Stick to these libraries: sklearn, numpy, pandas, scipy. If you explore beyond these, include installation code (!pip install xxx).
6.   Submit your solution as a single Ipython Notebook through Tabula, complete with comments explaining your code.
7.   Also, turn in a prediction file for the test data, formatted as prescribed.
��   Important:
No recycling old solutions, please! This year's dataset is a whole new game compared to previous years, demanding fresh answers.
Question No. 1: (Exploring data) [10% Marks]
Start by loading the training and test data. Once you have it ready, let's explore with these questions:
i.   Dataset Overview
a.   How many examples of each class are in the training set? And in the test set?
b.   Does this distribution of positive and negative examples signify any potential issues in terms of design of the machine learning solution and its evaluation? If so, please explain. 
ii.   Visual Data Exploration
a.   Pick 10 random objects from each class in the training data and display them using plt.matshow. Reshape the flattened 28x28 arrays for this. What patterns or characteristics do you notice?
b.   Do the same for 10 random objects from the test set. Are there any peculiarities in the data that might challenge your classifier's ability to generalize?
iii.   Choosing the Right Metric
Which performance metric would be best for this task (accuracy, AUC-ROC, AUC-PR, F1, Matthews correlation coefficient, mean squared error etc.)? Define each metric and discuss your reasoning for this choice.
iv.   Benchmarking a Random Classifier
Imagine a classifier that produces a random prediction score in the range [-1,+1] for a given input example. What accuracy would you expect it to achieve on both the training and test datasets? Show this through a coding experiment.
v.   Understanding AUC Metrics for Random Classifier
What would be the AUC-ROC and AUC-PR for a random classifier in this context? Again, support your answer with a code and discuss the consequences. 
Question No. 2: (Nearest Neighbor Classifier) [10% Marks]
Perform. 5-fold stratified   cross-validation (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) over the training   dataset using a k-nearest neighbour (kNN) classifier and answer the following questions:
i.   Start with a k = 5 nearest neighbour classifier. Define and calculate   the accuracy, balanced accuracy, AUC-ROC, AUC-PR, F1 and Matthews Correlation Coefficient for each fold using this classifier? Show code to demonstrate the results. Calculate the average and standard deviation for each metric across all folds and show these in a single table. As the KNN classifier in sklearn does not support decision_function, be代 写CS909/CS429 Data Mining 2025 Assignment 1: ClassificationPython
代做程序编程语言 sure to understand and use the predict_proba function for AUC-ROC and AUC-PR calculations or plotting. 
ii.   Plot the ROC and PR curves for one fold. What are your observations about the ROC and PR curves? What    part of the ROC curve is more important for this problem and why?
Question No. 3:    [20% Marks] Cross-validation of SVM and RFs
Use 5-fold stratified cross-validation over training data to choose an optimal classifier between:   SVMs (linear, polynomial kernels and Radial Basis Function Kernels) and Random Forest Classifiers. Be sure to tune the hyperparameters of each classifier type (C and kernel type and kernel hyper-parameters for SVMs, the number of trees, depth of trees etc. for the Random Forests etc). Report the cross validation results (mean and standard deviation of accuracy, balanced accuracy, AUC-ROC and AUC-PR across fold) of your best model. You may look into grid search as well as ways of pre-processing data (https://scikit-learn.org/stable/modules/preprocessing.html ) (e.g., mean-standard deviation or standard scaling or min-max scaling). 
i.   Write your strategy for selecting the optimal classifier. Show code to demonstrate the results for each classifier. 
ii.   Show the comparison of these classifiers in a single consolidated table. 
iii.   Plot the ROC curves of all classifiers on the same axes for easy comparison. 
iv.   Plot the PR curves of all classifier on the same axes for comparison. 
v.   Write your observations about the ROC and PR curves. 
Question No. 4 [20% Marks] PCA
i.   Reduce the number of dimensions of the training data using PCA to 2 and plot a scatter plot of the training data showing examples of each class in a different color. What are your observations about the data based on this plot?
ii.   Reduce the number of dimensions of the training and test data together using PCA to 2 and plot a scatter plot of the training and test data showing examples of each set in a different color (or marker style). What are your observations about the data based on this plot?
iii.   Plot the scree graph of PCA and find the number of dimensions that explain 95% variance in the training set. 
iv.   Reduce the number of dimensions of the data using PCA and perform. classification.    You may want to select different principal components for the classification (not necessarily the first few). What is the (optimal) cross-validation performance of a Kernelized SVM classification with PCA? Remember to perform. hyperparameter optimization! 
Question No. 5 Optimal Pipeline [20% Marks]
Develop an optimal pipeline for classification based on your analysis (Q1-Q4). You are free to use any tools or approaches at your disposal. However, no external data sources may be used. Describe your pipeline and report your outputs over the test data set. (You are required to submit your prediction file together with the assignment in a zip folder). Your prediction file should be a single column file containing the prediction score of the corresponding example in Xtest (be sure to have the same order as the order of the test examples in Xtest!). Your prediction file should be named by your student ID, e.g., u100011.csv. 
Question No. 6 Another classification problem [20% Marks]
Using the data given to you, consider an alternate classification problem in which the label of an example is based on whether it is a part of the training set (label = -1) or the test set (label = +1). Calculate the average and standard deviation of AUC-ROC using 5-fold stratified cross-validation for a classifier that is trained    to solve this prediction task. 
i.   What does the value of this AUC-ROC tell you about any differences between training and test sets? Show code for this analysis and clearly explain your conclusions with supporting evidence. 
ii.   How can you use this experiment to identify and eliminate any systematic differences between training and test sets?
iii.   Add random noise and random rotations to training set examples and then check if the AUC-ROC of this predictor changes. Clearly write and explain your observations.
   

         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
