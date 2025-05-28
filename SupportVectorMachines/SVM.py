# **Credit Card Fraud Detection with Decision Trees and SVM**
# Introduction

# Imagine that you work for a financial institution and part of your job is to build a model that predicts if a credit card transaction is fraudulent or not. 
# You can model the problem as a binary classification problem. A transaction belongs to the positive class (1) if it is a fraud, otherwise it belongs to the negative class (0).

# You have access to transactions that occured over a certain period of time. The majority of the transactions are normally legitimate and only a small fraction are non-legitimate. 
# Thus, typically you have access to a dataset that is highly unbalanced. This is also the case of the current dataset: only 492 transactions out of 284,807 
# are fraudulent (the positive class - the frauds - accounts for 0.172% of all transactions).

# This is a Kaggle dataset. You can find this "Credit Card Fraud Detection" dataset from the following link: Credit Card Fraud Detection.

# To train the model, you can use part of the input dataset, while the remaining data can be utilized to assess the quality of the trained model. First, 
# let's import the necessary libraries and download the dataset.


from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')


url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
# raw_data.to_csv('data.csv', index=False)  # Saves it locally

# Hədəf (target) dəyişənin siniflərini alırıq (0 və 1)
labels = raw_data.Class.unique()

# Hər sinifin neçə dəfə təkrarlandığını sayırıq
sizes = raw_data.Class.value_counts().values

# Pie chart çəkirik - balanssızlığı göstərir
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Hədəf dəyişənin siniflərinin faiz nisbəti')
# plt.show()  # əgər vizuallaşdırmaq istəsəniz bunu aktiv edin


# 'Class' ilə digər sütunlar arasındakı korrelyasiyanı hesablamaq
correlation_values = raw_data.corr()['Class'].drop('Class')

# Bar chart ilə vizuallaşdırmaq
correlation_values.plot(kind='barh', figsize=(10, 6))



# Z-normallaşdırma tətbiq edirik: ortalama = 0, standart sapma = 1
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])

# Numpy massivinə çeviririk
data_matrix = raw_data.values

# X: Giriş xüsusiyyətləri (Time sütunu xaric olmaqla 1-29)
X = data_matrix[:, 1:30]

# y: Çıxış/hədəf sinif (30-cu sütun: Class)
y = data_matrix[:, 30]

# L1 normallaşdırma tətbiq edirik
X = normalize(X, norm="l1")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Saxtakarlıq çox az olduğu üçün nümunələrə ağırlıq veririk
w_train = compute_sample_weight('balanced', y_train)




# # Decision Tree təsnifatçısı qururuq, maksimum dərinlik 4
# dt = DecisionTreeClassifier(max_depth=4, random_state=35)

# # Modeli öyrədirik
# dt.fit(X_train, y_train, sample_weight=w_train)

# # Test üçün ehtimallar çıxarırıq (1 sinfinin ehtimalı)
# y_pred_dt = dt.predict_proba(X_test)[:, 1]

# # ROC-AUC metriyi ilə qiymətləndirmə
# roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
# print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))






# SVM modeli balanslaşdırılmış sinif ağırlıqları ilə
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

# Modeli öyrədirik
svm.fit(X_train, y_train)

# Test üçün qərar funksiyası çıxarırıq
y_pred_svm = svm.decision_function(X_test)

# ROC-AUC ilə SVM modelini qiymətləndiririk
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))
