


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


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

df.count()


df = df.dropna()
df.info()


df.columns



df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()

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
    



#Excercise 1
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the 'Date' column
df = df.drop(columns=['Date'])




#Exercise 2 
X = df.drop(columns='RainTomorrow', axis=1)
y = df['RainTomorrow']




#Exerceise 3

y.value_counts()




#Exercise 4
# How often does it rain annually in the Melbourne area?

# Based on the counts, it rains on approximately X% of days in a year (e.g., if rain occurs on 30% of days, that means about 110 days per year on average).
# (Replace X% with your actual count from the dataset.)

# How accurate would you be if you just assumed it won't rain every day?

# If you always predicted “no rain”, your accuracy would be roughly equal to the percentage of days with no rain (e.g., around 70% accuracy if 70% of days are dry).

# This baseline accuracy shows that a naive model can be fairly accurate just by predicting the majority class.

# Is this a balanced dataset?

# No, this dataset is imbalanced because the number of no-rain days significantly exceeds the number of rain days.

# Imbalanced datasets can make model training challenging as models might be biased toward the majority class.

# Next steps?

# Use techniques to address imbalance, such as:

# Stratified splitting (to keep class distribution in train/test sets)

# Resampling methods (oversampling minority class or undersampling majority class)

# Use evaluation metrics beyond accuracy (e.g., precision, recall, F1-score)

# Experiment with models that handle imbalance better, like Random Forest or ensemble methods

# Feature engineering to improve model’s ability to distinguish rain vs no-rain days.




# Exercise 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)




# Exercise 6
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()



#Exercise 7
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)



# Exercise 8
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])



#Exercise 9
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))





# Exercise 10
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))




#Exercise 11
y_pred = grid_search.predict(X_test)



# Exercise 12

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#Exercise 13
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# Exercise 14
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_
 





 #Exercise 15

# Replace RandomForestClassifier with LogisticRegression in the pipeline
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Update the estimator in grid_search to the new pipeline
grid_search.estimator = pipeline

# Define the new param grid for LogisticRegression
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated grid_search with LogisticRegression
grid_search.fit(X_train, y_train)

# Make predictions on test set
y_pred = grid_search.predict(X_test)