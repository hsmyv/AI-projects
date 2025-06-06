# Practice Project: Titanic Survival Prediction
# Introduction
# Now that you have a feel for how to optimize your machine learning pipeline, let's practice with a real world dataset.
# You'll use cross validation and a hyperparameter grid search to optimize your machine learning pipeline.

# You will use the Titanic Survival Dataset to build a classification model to predict whether a passenger survived the sinking of the Titanic, based on attributes of each passenger in the data set.

# You'll start with building a Random Forest Classifier, then modify your pipeline to use a Logistic Regression estimator instead. You'll evaluate and compare your results.

# This lab will help prepare you for completing the Final Project.

# Objectives
# After completing this lab you will be able to:

# Use scikit-learn to build a model to solve a classification problem
# Implement a pipeline to combine your preprocessing steps with a machine learning model
# Interpret the results of your modelling
# Update your pipeline with a different machine learning model
# Compare the preformances of your classifiers


# !pip install numpy
# !pip install matplotlib
# !pip install pandas
# !pip install scikit-learn
# !pip install seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



# Titanic Passenger data set
# We'll be working with the Titanic passenger dataset to build a classification model to predict whether a passenger survied the sinking of the Titanic.
# Here is the data dictionary:




# Variable	                Definition
# survived	                survived? 0 = No, 1 = yes
# pclass	                Ticket class (int)
# sex	                    sex
# age	                    age in years
# sibsp	                    # of siblings / spouses aboard the Titanic
# parch	                    # of parents / children aboard the Titanic
# fare	                    Passenger fare
# embarked	                Port of Embarkation
# class	                    Ticket class (obj)
# who	                    man, woman, or child
# adult_male	            True/False
# alive	                    yes/no
# alone	                    yes/no





# Load the Titanic dataset using 

titanic = sns.load_dataset('titanic')
titanic.head()



# Select relevant features and the target
titanic.count()





# Features to drop
# deck has a lot of missing values so we'll drop it. age has quite a few missing values as well. 
# Although it could be, embarked and embark_town don't seem relevant so we'll drop them as well. 
# It's unclear what alive refers to so we'll ignore it.

# Target
# survived is our target class variable.

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]




# Exercise 1. How balanced are the classes?

y.value_counts()
# So about 38% of the passengers in the data set survived.  
# Because of this slight imbalance, 
# we should stratify the data when performing train/test split and for cross-validation.




# Exercise 2. Split the data into training and testing sets
# Don't forget to consider imbalance in the target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Define preprocessing transformers for numerical and categorical features
# Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features

numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# Define separate preprocessing pipelines for both feature types

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])



# Combine the transformers into a single column transformer
# We'll use the sklearn "column transformer" estimator to separately transform the features, 
# which will then concatenate the output as a single feature space, ready for input to a machine learning estimator.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])



# Create a model pipeline
# Now let's complete the model pipeline by combining the preprocessing with a Random Forest classifier


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# Define a parameter grid
# We'll use the grid in a cross validation search to optimize the model


param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}




# Perform grid search cross-validation and fit the best model to the training data
# Cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True)





# Exercise 3. Train the pipeline model

model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)


# Exercise 4. Get the model predictions from the grid search estimator on the unseen data
# Also print a classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))




# Exercise 5. Plot the confusion matrix
# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()









# Feature importances
# Let's figure out how to get the feature importances of our overall model. You'll need to know how to do this for your final project.
# First, to obtain the categorical feature importances, we have to work our way backward through the modelling pipeline to associate the feature importances with their one-hot encoded input features that were transformed from the original categorical features.

# We don't need to trace back through the pipeline for the numerical features, because we didn't transfrom them into new ones in any way.
# Remember, we went from categorical features to one-hot encoded features, using the 'cat' column transformer.

# Here's how you trace back through the trained model to access the one-hot encoded feature names:


model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)


# Notice how the one-hot encoded features are named - for example, sex was split into two boolean features indicating whether the sex is male or female.

# Great! Now let's get all of the feature importances and associate them with their transformed feature names.

feature_importances = model.best_estimator_['classifier'].feature_importances_

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))


# Display the feature importances in a bar plot
# Define a feature importance DataFrame, then plot it


importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis() 
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.show()

# Print test score 
test_score = model.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")





# Exercise 6. These are interesting results to consider.
# What can you say about these feature importances? Are they informative as is?

# The test set accuracy is somewhat satisfactory. However,regarding the feature impoirtances, it's crucially important to realize that there is most likely plenty of dependence amongst these variables, 
# and a more detailed modelling approach including correlation analysis is required to draw proper conclusions. For example, no doubt there is significant information shared by the variables `age`, `sex_male`, and `who_man`.


















# Try another model
# In practice you would want to try out different models and even revisit the data analysis to improve your model performance. Maybe you can engineer new features or impute missing values to be able to use more data.

# With Scikit-learn's powerful pipeline class, this is easy to do in a few steps. Let's update the pipeline and the parameter grid so we can train a Logistic Regression model and compare the performance of the two models.



# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
model.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

model.param_grid = param_grid

# Fit the updated pipeline with Logistic Regression
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)








# Exercise 7. Display the clasification report for the new model and compare the results to your previous model.

# print(classification_report(y_test, y_pred))
# All of the scores are slightly better for logistic regression than for random forest classification, although the differences are insignificant.




# Exercise 8. Display the confusion matrix for the new model and compare the results to your previous model.


# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()
# Again, the results show a slight improvement, with one more true positive and one more true negative.




# Extract the logistic regression feature coefficients and plot their magnitude in a bar chart
coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]

# Combine numerical and categorical feature names
numerical_feature_names = numerical_features
categorical_feature_names = (model.best_estimator_.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features)
                            )
feature_names = numerical_feature_names + list(categorical_feature_names)









# Exercise 9. Plot the feature coefficient magnitudes in a bar chart
# What's different about this chart than the feature importance chart for the Random Forest classifier?



# Create a DataFrame for the coefficients
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = model.best_estimator_.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")
# Although the performances of the two models are virtually identical, the features that are important to the two models are very different. 
# This suggests there must be more work to do to better grasp the actual feature importancdes. A smentioned above, 
# it's crucially important to realize that there is most likely plenty of dependence amongst these variables, 
# and a more detailed modelling approach including correlation analysis is required to draw proper conclusions. 
# For example, there is significant information implied between the variables who_man, who_woman, and who_child, because if a person is neither a man nor a woman, then they muct be a child.





# Congratulations! You've made it this far and are now fully equipped to take on your final project!