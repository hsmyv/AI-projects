# Lazımi kitabxanaların yüklənməsi
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model qurmaq üçün sklearn kitabxanalarından funksiyalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

# Lazımsız xəbərdarlıqları gizlədir
import warnings
warnings.filterwarnings('ignore')

# Dataseti online linkdən yükləyirik
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

# İlk 5 sətri göstəririk (kontrol məqsədi ilə)
data.head()

# Hədəf dəyişənin (NObeyesdad) sinif bölgüsünü göstəririk
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
# plt.show()

# ------------------------ PREPROCESSING ------------------------

# 1. Davamlı (ədədi) sütunların siyahısını çıxarırıq
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

# 2. Bu ədədi dəyərləri ortalaması 0, standart sapması 1 olacaq şəkildə standartlaşdırırıq
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])

# 3. Standartlaşdırılmış dəyərləri DataFrame formasına salırıq
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# 4. Bu yeni ədədi sütunları əsas datasetə əlavə edirik
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# 5. Kateqorik sütunları tapırıq (yəni mətn tipləri)
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Hədəf sütunu çıxarılır (onun üçün ayrıca işləyəcəyik)

# 6. One-hot encoding: Hər unikal dəyər üçün yeni sütun yaradılır (ilk dəyər atılır redundancy olmaması üçün)
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# 7. Encoded dəyərləri DataFrame formasına salırıq
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# 8. Bu yeni sütunları əsas datasetə əlavə edirik
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# 9. Hədəf dəyişəni (NObeyesdad) sayılara çeviririk (çoxsinifli prediksiya üçün)
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

# ------------------------ TRAIN-TEST BÖLGÜSÜ ------------------------

# Giriş və çıxışları ayırırıq
X = prepped_data.drop('NObeyesdad', axis=1)  # Giriş xüsusiyyətləri
y = prepped_data['NObeyesdad']              # Hədəf (çıxış) dəyərləri

# Məlumatları train və test hissələrinə bölürük (stratify - sinif bölgüsünü qoruyur)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------ LOGISTIC REGRESSION: OvA ------------------------

# One-vs-All strategiyası: Hər sinifə qarşı "digər hamısı" üçün ayrıca model qurulur
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Test datası üzərində proqnozlar
y_pred_ova = model_ova.predict(X_test)

# Dəqiqlik hesablanır
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova), 2)}%")

# ------------------------ LOGISTIC REGRESSION: OvO ------------------------

# One-vs-One strategiyası: Hər iki sinifin cüt-cüt müqayisəsi ilə çoxlu modellər qurulur
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Test datası üzərində proqnozlar
y_pred_ovo = model_ovo.predict(X_test)

# Dəqiqlik hesablanır
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo), 2)}%")






#Q1
# for test_size in [0.1, 0.3]:
#     # Məlumatı müxtəlif test ölçülərinə görə bölürük: 10% və 30% test üçün
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42, stratify=y
#     )

#     # Modeli öyrədirik
#     model_ova.fit(X_train, y_train)

#     # Test setində proqnozlaşdırırıq
#     y_pred = model_ova.predict(X_test)

#     # Nəticələri çap edirik
#     print(f"Test Size: {test_size}")
#     print("Accuracy:", accuracy_score(y_test, y_pred))




#Q2
# # Hansı xüsusiyyətlər (features) model üçün ən vacibdir?
# # Bunu modelin içindəki əmsallara (coefficients) baxaraq tapırıq.
# # Əgər bir feature-in əmsalı (koeffisienti) böyükdürsə, demək bu feature modelin qərarına çox təsir edir.

# # One-vs-All modeli üçün feature-ların əhəmiyyətini hesablayırıq
# # model_ova.coef_ => hər sinif üçün koeffisiyentlər (hər bir feature-in təsiri)
# feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)  # Ortalama mütləq dəyəri alırıq

# # Əhəmiyyət dəyərlərini bar chart şəklində vizuallaşdırırıq
# plt.barh(X.columns, feature_importance)
# plt.title("Feature Importance (One-vs-All)")
# plt.xlabel("Importance")
# plt.show()


# # One-vs-One modeli üçün feature əhəmiyyətini tapırıq
# # Hər binary classifier-in koeffisiyentlərini toplayırıq
# coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])

# # Bütün classifier-lərin ortalama mütləq koeffisiyentlərini hesablayırıq
# feature_importance = np.mean(np.abs(coefs), axis=0)

# # One-vs-One üçün əhəmiyyətləri bar chart ilə göstəririk
# plt.barh(X.columns, feature_importance)
# plt.title("Feature Importance (One-vs-One)")
# plt.xlabel("Importance")
# plt.show()





def obesity_risk_pipeline(data_path, test_size=0.2):
    # CSV faylını oxuyuruq
    data = pd.read_csv(data_path)

    # Rəqəmsal (float tipli) sütunları standartlaşdırırıq
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])

    # Standartlaşdırılmış məlumatları DataFrame-ə çeviririk
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

    # Standartlaşdırılmış sütunları orijinal datasetə əlavə edirik
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Obyekt tipli (yəni kateqorik) sütunları tapırıq
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Hədəf sütunu istisna edirik

    # One-hot encoding tətbiq edirik (dummy dəyişənlər)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

    # Kodlaşdırılmış kateqorikləri DataFrame-ə çeviririk
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Kateqorikləri çıxarıb, kodlaşdırılmışlarla əvəz edirik
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Target sütununu ədədi dəyərlərə çeviririk (label encoding)
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Giriş (X) və çıxış (y) dəyişənlərini ayırırıq
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']

    # Məlumatı train və test hissələrə bölürük
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Logistic Regression modeli qururuq
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)

    # Proqnozlaşdırırıq və dəqiqliyi ölçürük
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Funksiyanı çağırırıq
obesity_risk_pipeline(file_path, test_size=0.2)
