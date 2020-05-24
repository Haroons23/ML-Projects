THE PROBLEM: 

The Boston housing market is highly competitive, and we want to be the best real estate company in the area. To compete with our peers, we decide to leverage a few basic machine learning concepts to assist us and a client with finding the best selling price for their home. 

We have the Boston Housing dataset at our disposal. It contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. 
 

THE SOLUTION: We created a model to estimate the best selling price for our clients' homes.

1. Performed exploratory analysis:
- analyzed price distribution.
- analyzed correlation between features via heatmap. 
- analyzed marginal and joint distributions between features via pairplot.

2. Data Preprocessing:
- data was fairly clean to start with, no major outliers, or missing values.
- split data into training, test, and validation sets. Then scaled.

3. Modelling:
- performed linear regression.
- performed decision tree regression, and tuned 'max_depth' hyper parameter using gridsearch.
- tuned decision tree regressor performed best. 
- the model also performed well on the validation set. 

4. Next Steps:
- Get a larger dataset. 490 data points lead to 'okay' predictions but we ultimately need more. 
- More playing around with the data, looking for latent features...

