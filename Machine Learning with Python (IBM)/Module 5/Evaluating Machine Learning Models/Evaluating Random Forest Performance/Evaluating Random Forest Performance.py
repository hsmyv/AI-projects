# Evaluating Random Forest Performance
# Estimated time needed: 30 minutes

# Objectives
# After completing this lab, you will be able to:

# Implement and evaluate the performance of random forest regression models on real-world data
# Interpret various evaluation metrics and visualizations
# Describe the feature importances for a regression model
# Introduction
# In this lab, you will:

# Use the California Housing data set included in scikit-learn to predict the median house price based on various attributes
# Create a random forest regression model and evaluate its performance
# Investigate the feature importances for the model
# Your goal in this lab is not to find the best regressor - it is primarily intended for you to practice interpreting modelling results in the context of a real-world problem.

# Before you start, make sure that all the required libraries are installed. Execute the cell below for the same.




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew


# Load the California Housing data set


# Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target



# Print the description of the California Housing data set
print(data.DESCR)












# Exercise 1. Split the data into training and testing sets
# Reserve 20% of the data for evaluation

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

eda = pd.DataFrame(data=X_train)
eda.columns = data.feature_names
eda['MedHouseVal'] = y_train
eda.describe()










# Exercise 2. What range are most of the median house prices valued at?

# Considering the 25th to the 75th percentile range, most of the median house prices fall within $119,300 and $265,000.







# How are the median house prices distributed?

# Plot the distribution
plt.hist(1e5*y_train, bins=30, color='lightblue', edgecolor='black')
plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

# Evidently the distribution is skewed and there are quite a few clipped values at around $500,000.







# Model fitting and prediction
# Let's fit a random forest regression model to the data and use it to make median house price predicions. 
# Use the default parameters, which includes using 100 base estimators, or regression trees.


# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_regressor.predict(X_test)





# Estimate out-of-sample MAE, MSE, RMSE, and R²
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")














# Exercise 3. What do these statistics mean to you?
# How comfortable could you be with stopping here and communicating the results to the C-suite?


# The mean absolute error is $33,220.

# So, on average, predicted median house prices are off by $33k.

# Mean squared error is less intuitive to interpret, but is usually what is being minimized by the model fit.
# On the other hand, taking the square root of MSE yields a dollar value, here RMSE = $50,630.

# An R-squared score of 0.80 is not considered very high. It means the model explains about %80 of the variance in median house prices, although this interpretation can be misleading for compex data with nonlinear relationships, skewed values, and outliers. R-squard can still be useful for comparing models though.

# These statistics alone don't explain any details about the performance of the model. For example, where did the model do well or poorly?

# We aren't done yet!







# Plot Actual vs Predicted values


plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.show()










# Exercise 4. Plot the histogram of the residual errors (dollars)

# Also, print the mean and standard deviation of the residuals. Check for any patterns.

# Calculate the residual errors
residuals = 1e5*(y_test - y_pred_test)

# Plot the histogram of the residuals
plt.hist(residuals, bins=30, color='lightblue', edgecolor='black')
plt.title(f'Median House Value Prediction Residuals')
plt.xlabel('Median House Value Prediction Error ($)')
plt.ylabel('Frequency')
print('Average error = ' + str(int(np.mean(residuals))))
print('Standard deviation of error = ' + str(int(np.std(residuals))))


# The residuals are normally distributed with a very small average error and a standard deviation of about $50,000.











# Exercise 5. Plot the model residual errors by median house value.

# Sort the residuals by actual median house value before plotting the residuals.

# Create a DataFrame to make sorting easy
residuals_df = pd.DataFrame({
    'Actual': 1e5*y_test,
    'Residuals': residuals
})

# Sort the DataFrame by the actual target values
residuals_df = residuals_df.sort_values(by='Actual')

# Plot the residuals
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], marker='o', alpha=0.4,ec='k')
plt.title('Median House Value Prediciton Residuals Ordered by Actual Median Prices')
plt.xlabel('Actual Values (Sorted)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()









# Exercise 6. What trend can you infer from this residual plot?
# Although we saw a small average residual of only -$1400, you can see from this plot that the average error as a function of median house price is actually increasing from negative to positive values. 
# In other words, lower median prices tend to be overpredicted while higher median prices tend to be underpredicted.








# Exercise 7. Display the feature importances as a bar chart.
# Do you think these feature weights have practial significance? Are any of the features possibly sharing importance with other correlated features?

#Solution 1
# Feature importances
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

# Plot feature importances
plt.bar(range(X.shape[1]), importances[indices],  align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.show()



#Solution 2
# It makes sense that median incomes and house prices would be correlated, so it's not surprising that median income is the most important feature.
# I would think that location must be a significant factor.

# Since location is implied by two separate variables, latitude and longitude that share equal importances, we might speculate that location is really the second most important feature. This is because replacing latitude and longitude with a categorical location at an appropriate level of granularity (suburb, city, etc.) would likely have a combined lat/lng importance, which might outweigh average occupancy.

# Might average occupancy and average number of bedrooms be correlated?

# A proper analysis of the feature set would include a correlation matrix.















# Exercise 8. Some final thoughts to consider
# Will the skewness affect the performance of Random Forest regression?
# Does the clipping of median house prices above $500,000 bias the predictions?
# Also, do we need to standardize the data?


# Compared to linear regression, random forest regression is quite robust against outliers and skewed distributions. This is because random forest regression doesn't make any assumptions about the data distribution, 
# where linear regression performs best with normally distributed data.
# Standardizing the data isn't necessary like it is for distance-based algortihms like KNN or SVMs.

# Regarding the clipped vlaues, there is no variablilty in those values. Removing them in preprocessing might help the model to better explain the actual variance in the data.

# The clipped values can alsos bias the predictions. Also these clipped values can mislead evaluation metrics. As you've learned from this lab, it's crucially important for you to visualize your results.










# Summary
# Congratulations! You're ready to move on to your next lesson! In this lab, you learned how to implement and assess the performance of random forest regression models using real-world data. 
# You explored evaluation metrics and visualizations to understand the model's effectiveness and examined the feature importances in predicting outcomes. 
# Using the California Housing dataset, you built a random forest regression model to predict median house prices based on various attributes, then evaluated and interpreted the results to understand the factors influencing the predictions.