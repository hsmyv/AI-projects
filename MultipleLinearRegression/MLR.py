import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Dataset URL
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

# CSV faylını oxuyuruq
df = pd.read_csv(url)

# Məlumatdan təsadüfi 5 sətri göstərmək (yoxlama məqsədli)
df.sample(5)

# Sadəcə işimizə lazım olan sütunları saxlayırıq
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 
              'FUELTYPE','CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
              'FUELCONSUMPTION_COMB'], axis=1)

# Sütunlar arasındakı korrelyasiya matrisinə baxmaq üçün
df.corr()

# İlk 9 sətri göstər (məlumatın ümumi quruluşuna baxmaq üçün)
df.head(9)

# Datasetdəki dəyişənlər arasında nöqtə diaqramı matrisi (vizual analiz üçün)
# axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# for ax in axes.flatten():
#     ax.xaxis.label.set_rotation(90)
#     ax.yaxis.label.set_rotation(0)
#     ax.yaxis.label.set_ha('right')
# plt.tight_layout()
# plt.gcf().subplots_adjust(wspace=0, hspace=0)
# plt.show()

# Modelə giriş üçün müstəqil dəyişənlər (ENGINESIZE və FUELCONSUMPTION) və hədəf dəyişən (CO2EMISSIONS)
X = df.iloc[:, [0, 1]].to_numpy()  # ENGINESIZE, FUELCONSUMPTION
y = df.iloc[:, [2]].to_numpy()     # CO2EMISSIONS

# Dəyişənləri standartlaşdırırıq (ortalama = 0, standart sapma = 1)
# Bu, modelin sabit və daha dəqiq öyrənməsini təmin edir
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# Verilənləri təlim və test dəstlərinə bölürük (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Standartlaşdırılmış giriş məlumatının təsviri statistikası
pd.DataFrame(X_std).describe().round(2)

# Xətti reqressiya modelini təyin edirik
regressor = linear_model.LinearRegression()

# Modeli təlim məlumatı üzərində öyrədirik
regressor.fit(X_train, y_train)

# Modelin öyrəndiyi əmsallar və bias (kəsim) dəyəri
# Qeyd: Bunlar standartlaşdırılmış məkana aiddir
coef_ = regressor.coef_
intercept_ = regressor.intercept_

# Aşağıdakı addımlar nəticəni real (standartlaşdırılmamış) miqyasa qaytarmaq üçündür

# Giriş dəyişənlərinin orijinal ortalama və standart sapmaları
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# Standartlaşdırılmış əmsalları orijinal miqyasa qaytarırıq
coef_original = coef_ / std_devs_

# Kəsim dəyərini (intercept) də uyğun şəkildə düzəldirik
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

# Real dünya ölçüsündə modelin əmsallarını çap edirik
# print('Coefficients: ', coef_original)
# print('Intercept: ', intercept_original)



# Şərh:
# 17.86 (ENGINESIZE əmsalı):
# Mühərrikin ölçüsü (litrlə) 1 vahid artanda, digər faktorlar sabit qalmaqla, CO2 emissiyası təxminən 17.86 q/km artır.

# -5.02 (FUELCONSUMPTION əmsalı):
# FUELCONSUMPTION (kombinə edilmiş yanacaq sərfi) 1 vahid (L/100km) artanda, CO2 emissiyası təxminən 5.02 q/km azalır — bu bir qədər qəribə görsənə bilər, 
# çünki normalda daha çox yanacaq = daha çox emissiya gözlənilir. Bu, ya məlumatdakı konkret nümunələrdən, ya da daxil olan dəyişənlər arasındakı multikolinearlıqdan qaynaqlana bilər.

# Intercept (329.14):
# Hər iki dəyişən sıfır olsa belə, model CO2 emissiyasını 329.14 q/km olaraq təxmin edir. Bu, statistik olaraq bazal emissiya səviyyəsini göstərir.











# X1, X2 və y_test dəyişənlərini 3D qrafik üçün uyğun formata salırıq
# Əgər X_test 2 ölçülüdürsə, birinci və ikinci sütunları ayrıca götürürük
# Yox əgər tək ölçülüdürsə (yəni sadəcə bir dəyişən varsa), onda X2 üçün sıfırlardan ibarət array yaradılır
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

# Regressiya səthini çəkmək üçün meshgrid (xəritə tiplidir) yaradırıq
# Bu səth üzərində modelin proqnozlaşdırdığı y_surf dəyərləri hesablanacaq
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                               np.linspace(X2.min(), X2.max(), 100))

# Hər bir (x1, x2) cütü üçün proqnoz y dəyərləri hesablanır (regressiya tənliyi: y = a + b1*x1 + b2*x2)
y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf

# Modelin təlim görmüş regresiya obyektindən test datası üçün y_pred dəyərləri əldə edilir
# Bu dəyərlər real y_test ilə müqayisə olunaraq nöqtələrin təyyarədən yuxarı/aşağıda olması təyin olunur
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]

# 3D qrafik üçün sahə yaradırıq (20x8 ölçülü pəncərə)
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')  # 3D oxlarla subplot yaradılır

# Proqnoz səthindən yuxarıda qalan nöqtələr (real dəyərlər) bir rəngdə (məsələn, daha qabarıq)
# Aşağıda qalanlar isə fərqli (şəffaf) şəkildə göstərilir
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

# Regressiya səthi (təyyarə) çəkilir — şəffaf (alpha=0.21) və tünd rəngdə (qara)
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

# Qrafikin baxış bucağını təyin edirik (elev=10 dərəcəlik yuxarıdan baxış)
ax.view_init(elev=10)

# Əfsanə (izah), oxlar və qrafik başlığı əlavə olunur
ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])  # X oxunda tiqləri (dəyərləri) göstərmirik
ax.set_yticks([])  # Y oxunda tiqləri göstərmirik
ax.set_zticks([])  # Z oxunda tiqləri göstərmirik
ax.set_box_aspect(None, zoom=0.75)  # Qrafikin ölçülərinin nisbəti
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')  # X oxunun adı
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')  # Y oxunun adı
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')  # Z oxunun adı
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')  # Qrafikin başlığı
plt.tight_layout()  # Qrafikin çərçivədən çıxmamasını təmin edir
plt.show()  # Nəticəni göstərir






# 2D qrafiklərdə ENGINESIZE və FUELCONSUMPTION ilə CO2 emissiyasını göstəririk
plt.scatter(X_train[:,0], y_train,  color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# FUELCONSUMPTION ilə CO2 emissiyasını göstəririk
plt.scatter(X_train[:,1], y_train,  color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
plt.show()