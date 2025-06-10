from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
# raw_data.to_csv('yellow-tripdata.csv', index=False)


# Each row in the dataset represents a taxi trip. As shown above, each row has 13 variables. 
# One of the variables is tip_amount which will be the target variable. 
# Your objective will be to train a model that uses the other variables to predict the value of the tip_amount variable.

# To understand the dataset a little better, let us plot the correlation of the target variable against the input variables.




# 'tip_amount' ilə digər dəyişənlər arasındakı korrelyasiyanı hesablayırıq
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')

# Korrelyasiyanı üfüqi sütun şəklində vizuallaşdırırıq
correlation_values.plot(kind='barh', figsize=(10, 6))
#Bu hissədə tip_amount ilə digər sütunlar (məsələn: distance, fare_amount, passenger_count və s.) arasındakı əlaqə dərəcəsi vizuallaşdırılır.




# Nəticə dəyişəni (yəni modelin proqnozlaşdırmalı olduğu dəyər)
y = raw_data[['tip_amount']].values.astype('float32')
# y dəyişəni, yəni modelin proqnozlaşdırmalı olduğu dəyər (tip_amount) float32 tipinə çevrilir
# yəni modelin proqnozlaşdırmalı olduğu dəyər (tip_amount) float32 tipinə çevrilir



# tip_amount çıxarılır, çünki bu artıq target-dir
proc_data = raw_data.drop(['tip_amount'], axis=1)

# proc_data dəyişəni, yəni modelin giriş xüsusiyyətləri (digər sütunlar) 'tip_amount' sütunu çıxarılaraq yaradılır

# Qalan sütunlar giriş xüsusiyyətləri kimi istifadə olunacaq
X = proc_data.values
# X dəyişəni, yəni modelin giriş xüsusiyyətləri (digər sütunlar) numpy array formatına çevrilir




from sklearn.preprocessing import normalize

# Normalizasiya: bütün xüsusiyyətlər eyni ölçüdə olsun deyə (L1 norması ilə)
X = normalize(X, axis=1, norm='l1', copy=False)



# Verilənləri 70% təlim və 30% test olmaqla bölürük
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regression modelini qururuq
dt_reg = DecisionTreeRegressor(
    criterion='squared_error',  # MSE-yə əsaslanaraq ağacı böyüdür
    max_depth=8,                # Maksimum dərinlik 8-dir (çox dərin ağaclar overfitting edə bilər)
    random_state=35             # Təkrar nəticələr almaq üçün
)

# Modeli təlim edirik
dt_reg.fit(X_train, y_train)


# Model ilə test məlumatına əsasən proqnoz veririk
y_pred = dt_reg.predict(X_test)


# Mean Squared Error (Orta Kvadrat Səhv) — nə qədər yanıldığımızı ölçür
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

# R^2 skoru — 1-ə nə qədər yaxınsa, model o qədər yaxşıdır
r2_score = dt_reg.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score))



# MSE	 - Modelin orta səhvini göstərir (daha az daha yaxşıdır).
# R²	 - 1-ə nə qədər yaxındırsa, model bir o qədər güclüdür. 0 və ya mənfi ola bilərsə, model pisdir.







#Q2. Identify the top 3 features with the most effect on the `tip_amount`.



# ✅ Qısa cavab:
# Bu kod tip_amount (çay pulu) ilə ən çox əlaqəli olan (ən təsirli) 3 xüsusiyyəti tapır.

# 🔍 Ətraflı izah:
# 💡 correlation_values nədir?
# Əvvəldə bu sətir var idi:

#correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
# Bu, raw_data DataFrame-indəki bütün sütunların tip_amount ilə olan korrelyasiyasını hesablayır. Yəni:

# “Hansı sütunlar tip_amount dəyişkənini daha çox təsir edir?”


# | Sütun            | Korrelyasiya dəyəri |
# | ---------------- | ------------------- |
# | distance         | 0.45                |
# | fare\_amount     | 0.75                |
# | passenger\_count | -0.05               |


# 🔢 Bu nə edir?
# abs(correlation_values)
# Bütün korrelyasiya dəyərlərinin modulunu (müsbət versiyasını) alır, çünki həm müsbət, həm mənfi təsirlər vacibdir. Yəni -0.9 və 0.9 eyni dərəcədə güclüdür.




# .sort_values(ascending=False)
# Ən yüksəkdən aşağıya doğru düzür. Ən böyük təsiri olan xüsusiyyətlər yuxarıya çıxır.


# [:3]
# Ən yuxarıdakı 3 dəyəri seçir – yəni tip_amount-a ən çox təsir edən 3 xüsusiyyət.



# 📌 Misal nəticə:
# Tutaq ki, nəticə bu oldu:

# fare_amount       0.75  
# distance          0.60  
# duration_minutes  0.52

# Bu o deməkdir ki:
# fare_amount, distance, və duration_minutes dəyişkənləri çay puluna ən çox təsir edən xüsusiyyətlərdir.

# 🎯 Nəticə:
# Bu kod xətti, verilənlərdə ən vacib 3 xüsusiyyəti avtomatik seçmək üçün çox faydalıdır. Onları modelə daxil etməklə daha səmərəli təxminlər aparmaq olar.









#Q3.  Q3. Since we identified 4 features which are not correlated with the target variable, try removing these variables from the input set and see the effect on the 

# solution : raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)


# Gərəksiz və ya zəif əlaqəli sütunlar çıxarıldı:
# VendorID, payment_type, store_and_fwd_flag, improvement_surcharge kimi sütunlar çox vaxt:

# ya boşuna modelin kompleksliyini artırır

# ya da hədəf dəyişkənlə (tip_amount) zəif əlaqəsi olur.

# Bu cür sütunlar modelin öyrənməsini çaşdıra və ya səhv yönləndirə bilər









# Q4. Check the effect of decreasing the max_depth parameter to 4 on the 
