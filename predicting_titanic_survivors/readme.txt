PROBLEM

predict survival outcomes from the 1912 Titanic disaster based on each passenger’s features, such as sex and age.

SOLUTION:

1. Data Cleaning:
- identified columns to scale, encode, drop.
- imputed missing values with most common value for features missing very little data.
- created new groupings for prefix column. Went from 15 possible values to 8.
- dropped uninformative columns or ones with too many missing values. 

2. Exploratory Analysis:
- analyzed correlations between features via heatmap.
- analyzed marginal and joint distribtions between features via pairplot.
- analyzed victims and surivor counts based on sex and class (its best predictors).

3. Data Preprocessing:
- one hot encoded specific columns.
- used the KNN Imputer to impute for missing Age values. 
- scaled continuous features. 

4. Modelling:
- Used 4 different classification algorithms. Each one was tuned via grid search.
- gradient boosting classifier performed best: Accuracy of 86.54%, F1 score of 0.817.
- the 3 other algorithms had the same accuracy (81%) and slightly different F1 scores. 

5. Next Steps:
-  Further analyze ticket and cabin features. At current state, they have too many unique/ missing values to be useful. 