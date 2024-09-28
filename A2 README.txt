Project Title
---------------
Comparing the Effectiveness of K-Nearest Neighbors and Decision Trees in Predicting Whether a User Will Rate a Book as Top-Rated

Descriptions
---------------
This code analyses book ratings from three datasets (BX-Books.csv, BX-Ratings.csv, BX-Users.csv).
It prepreprocesses the data to clean any anomalies and enhance the data consistency.
Few statistical analysis has been done by visuaising various data distributions.
Machine learning models (K Nearest Neighbors and Decision Trees) are then built to predict rating classification. 
Various performance metrics are then computed as the base of comparison for the two models.

Prerequisites
---------------
Python 3 is required.
Libraries required are Pandas, NumPy, re, Seaborn, and Scikit-learn
The 3 datasets provided should be downloaded.

Output
---------------
This code will generate various plots to aid visualisation of various distributions.
Several performance metrics are also computed for both KNN and Decision Tree classifiers.

Conclusion
---------------
This code conducts a comparative analysis of two supervised machine learning models, K-Nearest Neighbours (KNN) and Decision Trees.
These comparison is done to figure out which model produces a more accurate book rating based on the reader’s age, author average ratings and average ratings of books published in the same year. 
This would assist bookstore managers to utilise the models to optimise inventory selection and produce more refined consumer recommendations. 