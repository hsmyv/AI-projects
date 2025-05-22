## Classification with Logistic Regression

### Scenario
# Assume that you are working for a telecommunications company which is concerned about the number of customers leaving their land-line business for cable competitors. 
# They need to understand who is more likely to leave the company.
    

# We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. 
# The data is relatively easy to understand, and you may uncover insights you can use immediately.
# Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
# This data set provides you information about customer preferences, services opted, personal details, etc. which helps you predict customer churn.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

# Dataset URL (IBM-dən Churn məlumatları)
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

# Dataseti oxuyuruq
churn_df = pd.read_csv(url)

# Sadəcə lazım olan sütunları saxlayırıq
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]

# 'churn' sütununu tam tipə (integer) çeviririk
churn_df['churn'] = churn_df['churn'].astype('int')

# X - xüsusiyyətləri (feature) təyin edirik
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  # İlk 5 sətri göstərə bilərik

# y - hədəf dəyişəni (churn olub-olmaması)
y = np.asarray(churn_df['churn'])
y[0:5]  # İlk 5 hədəfi göstərə bilərik

# X məlumatlarını normalizasiya edirik (orta = 0, standart sapma = 1)
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]  # İlk 5 normalizasiya olunmuş sətri göstərə bilərik

# Məlumatı train və test hissələrinə ayırırıq (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Logistic Regression modelini qururuq və öyrədirik
LR = LogisticRegression().fit(X_train, y_train)

# Test məlumatları üzərində proqnoz (təyin) edirik
yhat = LR.predict(X_test)
yhat[:10]  # İlk 10 proqnoz nəticəsini göstərə bilərik

# Test məlumatları üçün ehtimalları proqnozlaşdırırıq
yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]  # İlk 10 ehtimal nəticəsini göstərə bilərik

# Modelin hər bir xüsusiyyət üçün koeffisentlərini əldə edirik
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])

# Bu koeffisentləri çubuq diagramı ilə vizuallaşdırırıq
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Modelin log-loss dəyərini hesablayırıq
loss = log_loss(y_test, yhat_prob)  # Kiçik dəyər daha yaxşıdır — modelin performansını qiymətləndirmək üçün istifadə olunur

print("Log Loss: ", loss)
