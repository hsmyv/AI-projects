# Lab: K-Nearest Neighbors Classifier


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Təsəvvür edin ki, bir telekommunikasiya şirkəti öz müştərilərini xidmət istifadəsi modellərinə görə dörd qrupa ayırıb. 
# Əgər demoqrafik məlumatlar (məsələn: bölgə, yaş, ailə vəziyyəti) bu qrupları proqnozlaşdırmaq üçün istifadə oluna bilirsə, 
# şirkət potensial müştərilər üçün fərdi təkliflər hazırlaya bilər.

# Bu, klassifikasiya (təsnifat) problemdir.
# Yəni, bizə verilmiş etiketli məlumatlar əsasında yeni və ya naməlum bir nümunənin hansı qrupa aid olduğunu proqnozlaşdıracaq bir model qurmalıyıq.

# Biz bu nümunədə demoqrafik məlumatlardan istifadə edərək, istifadəçi xidmət qrupunu təxmin etməyə çalışacağıq.

# "custcat" adlı hədəf (target) sütunu dörd mümkün xidmət kateqoriyasına malikdir:
# 1. Basic Service (Əsas Xidmət)
# 2. E-Service (Elektron Xidmət)
# 3. Plus Service (Əlavə Xidmət)
# 4. Total Service (Tam Xidmət)

# Məqsədimiz: Yeni istifadəçi üçün bu dörd kateqoriyadan hansına aid olduğunu təxmin edən klassifikasiya modeli qurmaqdır.
# Burada istifadə etdiyimiz model "K ən yaxın qonşular" (KNN) adlanır.

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

# Datasetdəki etiketlərin (yəni müştəri kateqoriyalarının) sayını yoxlayaq
df['custcat'].value_counts()

# Atributlar (sütunlar) arasındakı əlaqəni göstərmək üçün korrelyasiya matrisini hesablayırıq
correlation_matrix = df.corr()

# Korrelyasiya matrisini istilik xəritəsi (heatmap) ilə göstəririk
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Hədəf dəyişən (custcat) ilə digər dəyişənlər arasındakı əlaqəni ölçürük və azalan sıralayırıq
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)

print("Hedef deyishen ile olan korrelyasiya:")
print(correlation_values)

# Məlumat dəstini xüsusiyyətlər (X) və hədəf dəyişən (y) olaraq ayırırıq
X = df.drop('custcat',axis=1)
y = df['custcat']

# Xüsusiyyətləri standartlaşdırırıq (ortalamanı çıxarıb vahid dispersiyaya gətiririk)
X_norm = StandardScaler().fit_transform(X)

# Məlumatları 80% təlim, 20% test olaraq ayırırıq
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Modeli qurmaq üçün k=3 təyin edirik
k = 3
# Modeli təlim etdiririk və proqnozlaşdırırıq
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)
yhat = knn_model.predict(X_test)

# Test dəstindəki düzgünlük (accuracy) nəticəsini çap edirik
print("Accuracy (Accuracy): ", accuracy_score(y_test, yhat))








# ================================
# Tapşırıq 1
# Eyni modeli yenidən qur, bu dəfə k=6 istifadə et

# k = 6
# knn_model_6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# yhat6 = knn_model_6.predict(X_test)
# print("Accuracy (Accuracy): ", accuracy_score(y_test, yhat6))

# # Müxtəlif k dəyərləri üçün modeli sınaqdan keçiririk və nəticələri müqayisə edirik
# Ks = 10
# acc = np.zeros((Ks))        # Hər k üçün dəqiqlik (accuracy) saxlanacaq massiv
# std_acc = np.zeros((Ks))    # Hər k üçün standart sapma saxlanacaq massiv

# for n in range(1,Ks+1):
#     # Modeli qur və nəticəni proqnozlaşdır
#     knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
#     yhat = knn_model_n.predict(X_test)
#     acc[n-1] = accuracy_score(y_test, yhat)  # Dəqiqliyi qeyd et
#     std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])  # Standart sapmanı hesabla

# # K-nin fərqli dəyərləri üçün modelin dəqiqliyini vizual şəkildə göstər
# plt.plot(range(1,Ks+1),acc,'g')
# plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Dəqiqlik', 'Standart Sapma'))
# plt.ylabel('Modelin dəqiqliyi')
# plt.xlabel('Qonşu sayı (K)')
# plt.tight_layout()
# plt.show()

# # Ən yüksək dəqiqliyin əldə edildiyi K dəyərini çap et
# print("Ən yaxşı dəqiqlik:", acc.max(), " ilə əldə olundu, k =", acc.argmax()+1)











# Exercise 2
# Ən yaxşı k dəyərini tapmaq üçün 30 fərqli k dəyəri ilə təlim aparılır

Ks_30 = 30  # 1-dən 30-a qədər olan k dəyərləri üçün
acc_30 = np.zeros((Ks_30))  # Hər k üçün dəqiqlik dəyərləri saxlanılır
std_acc_30 = np.zeros((Ks_30))  # Standart sapma saxlanılır

for n in range(1, Ks_30 + 1):
    # Modeli qur və təlim et
    knn_model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = knn_model.predict(X_test)
    acc_30[n - 1] = accuracy_score(y_test, yhat)  # Dəqiqliyi yadda saxla
    std_acc_30[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])  # Standart sapmanı hesabla

# Ən yaxşı nəticəni verən k dəyəri və dəqiqliyi
best_k_30 = acc_30.argmax() + 1  # .argmax() 0-dan başladığı üçün +1 edilir
best_acc_30 = acc_30.max()

print(f"1-den 30-a qeder en yaxshi netice {best_acc_30:.4f} deqiqlikle k = {best_k_30}-de elde olundu.")


# Ən yaxşı k dəyərini tapmaq üçün bu dəfə 100 fərqli k dəyəri ilə təlim aparılır

Ks_100 = 100  # 1-dən 100-ə qədər olan k dəyərləri üçün
acc_100 = np.zeros((Ks_100))
std_acc_100 = np.zeros((Ks_100))

for n in range(1, Ks_100 + 1):
    knn_model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = knn_model.predict(X_test)
    acc_100[n - 1] = accuracy_score(y_test, yhat)
    std_acc_100[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

best_k_100 = acc_100.argmax() + 1
best_acc_100 = acc_100.max()

print(f"1-den 100-e qeder en yaxsi netice {best_acc_100:.4f} deqiqlikle k = {best_k_100}-de elde olundu.")

# Vizual olaraq 100 k üçün nəticəni göstərmək üçün qrafik
plt.figure(figsize=(12, 6))
plt.plot(range(1, Ks_100 + 1), acc_100, 'b')
plt.fill_between(range(1, Ks_100 + 1), acc_100 - std_acc_100, acc_100 + std_acc_100, color='b', alpha=0.1)
plt.xlabel('K deyeri')
plt.ylabel('Modelin deqiqliyi')
plt.title('1-den 100-e qeder K deyerlerine gore deqiqlik')
plt.grid(True)
plt.show()






# Exercise 3
# Plot the variation of the accuracy score for the training set for 100 value of Ks.

# Ks =100
# acc = np.zeros((Ks-1))
# std_acc = np.zeros((Ks-1))
# for n in range(1,Ks):
#     #Train Model and Predict  
#     knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
#     yhat = knn_model_n.predict(X_train)
#     acc[n-1] = accuracy_score(y_train, yhat)
#     std_acc[n-1] = np.std(yhat==y_train)/np.sqrt(yhat.shape[0])

# plt.plot(range(1,Ks),acc,'g')
# plt.fill_between(range(1,Ks),acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Accuracy value', 'Standard Deviation'))
# plt.ylabel('Model Accuracy')
# plt.xlabel('Number of Neighbors (K)')
# plt.tight_layout()
# plt.show()









# Exercise 4
# Can you justify why the model performance on training data is deteriorating with increase in the value of k?


# Click here for the solution
# When k is small (e.g., k=1), the model is highly sensitive to the individual points in the dataset. The prediction for each point is based on its closest neighbor, 
# which can lead to highly specific and flexible boundaries. This leads to overfitting on the training data, meaning the model will perform very well on the training set,
# potentially achieving 100% accuracy. However, it may generalize poorly to unseen data. When k is large, the model starts to take into account more neighbors when making predictions. 
# This has two main consequences:

# 1.Smoothing of the Decision Boundary: The decision boundary becomes smoother, which means the model is less sensitive to the noise or fluctuations in the training data.
# 2.Less Specific Predictions: With a larger k, the model considers more neighbors and therefore makes more generalized predictions, which can lead to fewer instances being classified perfectly.
# As a result, the model starts to become less flexible, and its ability to memorize the training data (which can lead to perfect accuracy with small k) is reduced.





# Exercise 5
# We can see that even the with the optimum values, the KNN model is not performing that well on the given data set. Can you think of the possible reasons for this?

# Click here for the solution
# The weak performance on the model can be due to multiple reasons. 1. The KNN model relies entirely on the raw feature space at inference time. 
# If the features do no provide clear boundaries between classes, KNN model cannot compensate through optimization or feature transformation. 
# 2. For a high number of weakly correlated features, the number of dimensions increases, the distance between points tend to become more uniform, reducing the discriminative power of KNN. 
# 3. The algorithm treats all features equally when computing distances. Hence, weakly correalted features can introduce noise or irrelevant variations in the feature space making it harder for KNN to find meaningful neighbours.