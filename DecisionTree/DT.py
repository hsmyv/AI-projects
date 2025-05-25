# Decision Trees
# Develop a classification model using Decision Tree Algorithm
# Apply Decision Tree classification on a real world dataset.

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')

# About the dataset
# Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. 
# During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug C, Drug X and Drug Y.

# Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. 
# The features of this dataset are the Age, Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to.

# It is a sample of a multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict 
# the class of an unknown patient or to prescribe a drug to a new patient.


# Dataset-in internet üzərindən oxunması
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

# Kategorik dəyişənləri rəqəmlərə çeviririk (Label Encoding)
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])           # Kişi/qadın → 0/1
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])             # Blood Pressure (Low/Normal/High) → 0/1/2
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) # Cholesterol (Normal/High) → 0/1

# Dataset-də boş dəyər (missing value) olub-olmadığını yoxlayırıq
print(my_data.isnull().sum())  # Heç bir null dəyər olmadığı gözlənilir

# Burada Drug sütunundakı dərman adları rəqəmlərlə əvəz olunur ki, korrelyasiya və analizlər üçün asanlıq olsun. Amma bu sütun model üçün istifadə olunmur, sadəcə analiz üçündür.
custom_map = {'drugA':0, 'drugB':1, 'drugC':2, 'drugX':3, 'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

# Məqsəd dəyişən ilə digər dəyişənlər arasında korrelyasiya (əlaqə) hesablanır
#Burada Drug_num ilə digər sütunlar arasında əlaqə dərəcəsi hesablanır. 1-ə yaxınsa güclü əlaqə, 0-a yaxınsa zəif deməkdir.
print(my_data.drop('Drug', axis=1).corr()['Drug_num'])

# Hər dərman sinfinin sayını göstərən çubuq qrafiki
category_counts = my_data['Drug'].value_counts()
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Dərman')
plt.ylabel('Say')
plt.title('Dərman Kateqoriyalarının Paylanması')
plt.xticks(rotation=45)  # Ehtiyac olarsa, adları fırladır
plt.show()

# Modelə hazırlıq — hədəf dəyişən (y) və atributlar (X)
# y → proqnoz ediləcək dəyişən (Drug).

# X → təlim üçün istifadə ediləcək atributlar (Age, Sex, BP və s.).

# Drug_num burada istifadə edilmir çünki modelə kateqoriyaları olduğu kimi ötürürük.
y = my_data['Drug']  # Kateqorial hədəf (Drug)
X = my_data.drop(['Drug', 'Drug_num'], axis=1)  # Drug_num korrelyasiya üçün idi, modeldə istifadə edilmir

# Məlumatı train və test hissələrinə bölürük
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# Decision Tree modelini qururuq (entropy kriteriyası ilə, max_depth=4)
# criterion="entropy" → məlumatın qeyri-müəyyənliyini ölçür.
# max_depth=4 → ağacın maksimal dərinliyini 4-lə məhdudlaşdırır (overfitting qarşısı üçün).
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

# Test məlumatları üzərində proqnoz veririk
tree_predictions = drugTree.predict(X_testset)

# Nəticəni qiymətləndiririk
print("Decision Tree modelinin dəqiqliyi: ", metrics.accuracy_score(y_testset, tree_predictions))

# Decision Tree-ni vizual olaraq göstəririk
plot_tree(drugTree, feature_names=X.columns, class_names=drugTree.classes_, filled=True)
plt.show()