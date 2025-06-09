# Final Project: Building a Rainfall Prediction Classifier
# Estimated time needed: 60 minutes

# Objectives
# After completing this lab you will be able to:

# Explore and perform feature engineering on a real-world data set
# Build a classifier pipeline and optimize it using grid search cross validation
# Evaluate your model by interpreting various performance metrics and visualizations
# Implement a different classifier by updating your pipeline
# Use an appropriate set of parameters to search over in each case
# Instruction(s)
# After completing the Notebook:

# Download the notebook using File > Download.
# This notebook will be then graded using AI grader in the subsequent section.
# Copy/Paste your markdown responses in the subsequent AI Mark assignment.



# About The Dataset
# The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from http://www.bom.gov.au/climate/dwo/.

# The dataset you'll use in this project was downloaded from Kaggle at https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/
# Column definitions were gathered from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

# The dataset contains observations of weather metrics for each day from 2008 to 2017, and includes the following fields:

# Field	    Description	    Unit	    Type
# Date	Date of the Observation in YYYY-MM-DD	Date	object
# Location	Location of the Observation	Location	object
# MinTemp	Minimum temperature	Celsius	float
# MaxTemp	Maximum temperature	Celsius	float
# Rainfall	Amount of rainfall	Millimeters	float
# Evaporation	Amount of evaporation	Millimeters	float
# Sunshine	Amount of bright sunshine	hours	float
# WindGustDir	Direction of the strongest gust	Compass Points	object
# WindGustSpeed	Speed of the strongest gust	Kilometers/Hour	object
# WindDir9am	Wind direction averaged over 10 minutes prior to 9am	Compass Points	object
# WindDir3pm	Wind direction averaged over 10 minutes prior to 3pm	Compass Points	object
# WindSpeed9am	Wind speed averaged over 10 minutes prior to 9am	Kilometers/Hour	float
# WindSpeed3pm	Wind speed averaged over 10 minutes prior to 3pm	Kilometers/Hour	float
# Humidity9am	Humidity at 9am	Percent	float
# Humidity3pm	Humidity at 3pm	Percent	float
# Pressure9am	Atmospheric pressure reduced to mean sea level at 9am	Hectopascal	float
# Pressure3pm	Atmospheric pressure reduced to mean sea level at 3pm	Hectopascal	float
# Cloud9am	Fraction of the sky obscured by cloud at 9am	Eights	float
# Cloud3pm	Fraction of the sky obscured by cloud at 3pm	Eights	float
# Temp9am	Temperature at 9am	Celsius	float
# Temp3pm	Temperature at 3pm	Celsius	float
# RainToday	If there was at least 1mm of rain today	Yes/No	object
# RainTomorrow	If there is at least 1mm of rain tomorrow	Yes/No	object




import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# Load the data
# Execute the following cells to load the dataset as a pandas dataframe.


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

df.count()


# Sunshine and cloud cover seem like important features, but they have a lot of missing values, far too many to impute their missing values.

# Drop all rows with missing values
# To try to keep things simple we'll drop rows with missing values and see what's left


df = df.dropna()
df.info()

# Since we still have 56k observations left after dropping missing values, we may not need to impute any missing values.
# Let's see how we do.


df.columns


# Data leakage considerations
# Consider the descriptions above for the columns in the data set. Are there any practical limitations to being able to predict whether it will rain tomorrow given the available data?

# Points to note - 1
# List some of the features that would be inefficient in predicting tomorrow's rainfall. There will be a question in the quiz that follows based on this observation.


# Consider features that rely on the entire duration of today for their evaluation.

# If we adjust our approach and aim to predict today’s rainfall using historical weather data up to and including yesterday, then we can legitimately utilize all of the available features. This shift would be particularly useful for practical applications, such as deciding whether you will bike to work today.

# With this new target, we should update the names of the rain columns accordingly to avoid confusion.

df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })




# Data Granularity
# Would the weather patterns have the same predictability in vastly different locations in Australia? I would think not.
# The chance of rain in one location can be much higher than in another. Using all of the locations requires a more complex model as it needs to adapt to local weather patterns.
# Let's see how many observations we have for each location, and see if we can reduce our attention to a smaller region.

# Location selection
# You could do some research to group cities in the Location column by distance, which I've done for you behind the scenes.
# I found that Watsonia is only 15 km from Melbourne, and the Melbourne Airport is only 18 km from Melbourne.
# Let's group these three locations together and use only their weather data to build our localized prediction model.
# Because there might still be some slight variations in the weather patterns we'll keep Location as a categorical variable.


df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()



# We still have 7557 records, which should be enough to build a reasonably good model.
# You could always gather more data if needed by partioning the data into similar locations or simplyby updating it from the source to include a larger time frame.

# Extracting a seasonality feature
# Now consider the Date column. We expect the weather patterns to be seasonal, having different predictablitiy levels in winter and summer for example.
# There may be some variation with Year as well, but we'll leave that out for now. Let's engineer a Season feature from Date and drop Date afterward, since it is most likely less informative than season. An easy way to do this is to define a function that assigns seasons to given months, then use that function to transform the Date column.

# Create a function to map dates to seasons



def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'
    











# Exercise 1: Map the dates to seasons and drop the Date column
# Complete the code:

# # Convert the 'Date' column to datetime format
# df['Date'] = pd.to_datetime(...)

# # Apply the function to the 'Date' column
# df['Season'] = df['Date'].apply(date_to_season)

# df=df.drop(columns=...)
# df

# Write your response.












# Looks like we have a good set of features to work with.

# Let's go ahead and build our model.

# But wait, let's take a look at how well balanced our target is.

# Exercise 2. Define the feature and target dataframes
# Complete the followng code:

# X = df.drop(columns='...', axis=1)
# y = df['...']
# Write your response.






# Exercise 3. How balanced are the classes?
# Display the counts of each class.

# Complete the following code:

# ... .value_counts()

# Write your response.






# Exercise 4. What can you conclude from these counts?
# How often does it rain annualy in the Melbourne area?
# How accurate would you be if you just assumed it won't rain every day?
# Is this a balanced dataset?
# Next steps?










# Exercise 5. Split data into training and test sets, ensuring target stratification
# Complete the followng code:

# X_train, X_test, y_train, y_test = train_test_split(..., ..., test_size=0.2, stratify=..., random_state=42)
# Write your response.











# Define preprocessing transformers for numerical and categorical features
# Exercise 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features
# Complete the followng code:

# numeric_features = X_train.select_dtypes(include=['...']).columns.tolist()  
# categorical_features = X_train.select_dtypes(include=['...', 'category']).columns.tolist()

# Write your response.








# Define separate transformers for both feature types and combine them into a single preprocessing transformer


# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])






# Exercise 7. Combine the transformers into a single preprocessing column transformer
# Complete the followng code:

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, ...),
#         ('cat', categorical_transformer, ...)
#     ]
# )

# Write your response.





# Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier
# Complete the following code:

# pipeline = Pipeline(steps=[
#     ('preprocessor', ...),
#     ('...', RandomForestClassifier(random_state=42))
# ])
# Write your response.







# Define a parameter grid to use in a cross validation grid search model optimizer

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Pipeline usage in crossvalidation
# Recall that the pipeline is repeatedly used within the crossvalidation by fitting on each internal training fold and predicting on its corresponding validation fold



# Perform grid search cross-validation and fit the best model to the training data
# Select a cross-validation method, ensuring target stratification during validation
cv = StratifiedKFold(n_splits=5, shuffle=True)







# Exercise 9. Instantiate and fit GridSearchCV to the pipeline
# Complete the followng code:

# grid_search = GridSearchCV(..., param_grid, cv=..., scoring='accuracy', verbose=2)  
# grid_search.fit(..., ...)
### Write your response.




## Print the best parameters and best crossvalidation score
# print("\nBest parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))







# Exercise 10. Display your model's estimated score
# Complete the followng code:

# test_score = grid_search.score(..., ...)  
# print("Test set score: {:.2f}".format(test_score))

## Write your response.
# So we have a reasonably accurate classifer, which is expected to correctly predict about 84% of the time whether it will rain today in the Melbourne area.
# But careful here. Let's take a deeper look at the results.

# The best model is stored within the gridsearch object.





# Exercise 11. Get the model predictions from the grid search estimator on the unseen data
# Complete the followng code:

# y_pred = grid_search.predict(...)
### Write your response.







# Exercise 12. Print the classification report
# Complete the followng code:

# print("\nClassification Report:")
# print(...(y_test, y_pred))
## Write your response.







# Exercise 13. Plot the confusion matrix
# Complete the followng code:

# conf_matrix = ...(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=...)
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()
## Write your response.



# Let's consider wether the results indicate a good predictor of rainfall.

# Points to note - 2
# What is the true positive rate? There will be a question on this in the assignment that follows
# Consider the confusion matrix or the classification report and claculate the true positve rate given the information..








# Feature importances
# Recall that to obtain the categorical feature importances, we have to work our way backward through the modelling pipeline to associate the feature importances with their original input variables, not the one-hot encoded ones. We don't need to do this for the numeric variables because we didn't modify their names in any way.
# Remember we went from categorical features to one-hot encoded features, using the 'cat' column transformer.

# Let's get all of the feature importances and associate them with their transformed features


# Exercise 14. Extract the feature importances
# Complete the followng code:

# feature_importances = grid_search.best_estimator_['classifier']. ...
## Write your response.

# Now let's extract the feature importances and plot them as a bar graph.


# Combine numeric and categorical feature names
# feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
#                                         .named_transformers_['cat']
#                                         .named_steps['onehot']
#                                         .get_feature_names_out(categorical_features))

# feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# importance_df = pd.DataFrame({'Feature': feature_names,
#                               'Importance': feature_importances
#                              }).sort_values(by='Importance', ascending=False)

# N = 20  # Change this number to display more or fewer features
# top_features = importance_df.head(N)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
# plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
# plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
# plt.xlabel('Importance Score')
# plt.show()













# Point to note - 3
# Identify the most important feature for predicting whether it will rain based on the feature importance bar graph. There will be a question on this in the assignment that follows.

# Try another model
# Some thoughts.
# In practice you would want to try out different models and even revisit the data analysis to improve your model's performance. 
# Maybe you can engineer better features, drop irrelevant or redundant ones, project your data onto a dimensional feature space, or impute missing values to be able to use more data. 
# You can also try a larger set of parameters to define you search grid, or even engineer new features using cluster analysis. You can even include the clustering algorithm's hyperparameters in your search grid!

# With Scikit-learn's powerful pipeline and GridSearchCV classes, this is easy to do in a few steps.

# Exercise 15. Update the pipeline and the parameter grid
# Let's update the pipeline and the parameter grid and train a Logistic Regression model and compare the performance of the two models. You'll need to replace the clasifier with LogisticRegression. We have supplied the parameter grid for you.






# # Replace RandomForestClassifier with LogisticRegression
# pipeline.set_params(...=LogisticRegression(random_state=42))

# # update the model's estimator to use the new pipeline
# grid_search.estimator = ...

# # Define a new grid with Logistic Regression parameters
# param_grid = {
#     # 'classifier__n_estimators': [50, 100],
#     # 'classifier__max_depth': [None, 10, 20],
#     # 'classifier__min_samples_split': [2, 5],
#     'classifier__solver' : ['liblinear'],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__class_weight' : [None, 'balanced']
# }

# grid_search.param_grid = ...

# # Fit the updated pipeline with LogisticRegression
# model.fit(..., ...)

# # Make predictions
# y_pred = model.predict(X_test)
## Write your response









# Compare the results to your previous model.
# Display the clasification report and the confusion matrix for the new model and compare your results with the previous model.

# print(classification_report(y_test, y_pred))

# # Generate the confusion matrix 
# conf_matrix = confusion_matrix(y_test, y_pred)

# plt.figure()
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# # Set the title and labels
# plt.title('Titanic Classification Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')

# # Show the plot
# plt.tight_layout()
# plt.show()




# What can you conclude about the model performances?

# Points to note - 4
# Compare the accuracy and true positive rate of rainfall predictions between the LogisticRegression model and the RandomForestClassifier model.

# Note: Make sure to provide the answer in the form of a list using either bullets or numbers.

# There will be a question on this in the assignment that follows.



# Compare the accuracy percentages of both the classifiers.

# Provide the details of the number of correct predictions.

# Provide the true positive rate of LogisticRegression Classifier.


# Congratulations! You've made it the end of your final project!
# Well done! You now have some great tools to use for tackling complex real-world problems with machine learning.