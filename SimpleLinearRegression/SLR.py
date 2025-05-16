import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df=pd.read_csv(url)
# print(df.sample(5))

# print(df.describe())

# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.sample(9))



# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# viz.hist()
# plt.show()



# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.xlim(0,27)
# plt.show()


# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("CYLINDERS")
# plt.ylabel("CO2 Emission")
# plt.show()





# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# X = cdf.ENGINESIZE.to_numpy()
# y = cdf.CO2EMISSIONS.to_numpy()


# X = X.reshape(-1, 1)

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# print(type(X_train))
# print(np.shape(X_train))
# print(np.shape(X_test))






# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# X = cdf.ENGINESIZE.to_numpy()
# y = cdf.CO2EMISSIONS.to_numpy()
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# regressor = linear_model.LinearRegression()

# regressor.fit(X_train.reshape(-1, 1), y_train)

# print ('Coefficients: ', regressor.coef_[0])
# print ('Intercept: ',regressor.intercept_)

# plt.scatter(X_train, y_train, color='blue')
# plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()










# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# X = cdf.ENGINESIZE.to_numpy()
# y = cdf.CO2EMISSIONS.to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# regressor = linear_model.LinearRegression()
# regressor.fit(X_train.reshape(-1, 1), y_train)

# y_test_ = regressor.predict(X_test.reshape(-1,1))



# #Bu metrik real nəticə ilə modelin proqnoz verdiyi nəticə arasındakı orta "mütləq" fərqi ölçür.
# #Yəni heç bir fərqi kvadratlamır, sadəcə məsafəyə baxır (| fərq |).
# # Sadə izah: Proqnozlar gerçək dəyərlərdən ortalama nə qədər uzaqdadır?
# print("Mean absolute error: %.2f"  % mean_absolute_error(y_test_, y_test))


# #Mean Squared Error (MSE) — Orta Kvadrat Səhv
# print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))

# #Root Mean Squared Error (RMSE) — Orta Kvadrat Səhvin Kökü
# #Bu sadəcə MSE-nin kvadrat köküdür və yenidən ölçü vahidini geri qaytarır.
# print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))

# #R² Score (Determination Coefficient) — Açıklama Əmsalı
# #Bu modelin nə qədər yaxşı açıqladığını göstərir. 1.0 ideal nəticədir.
# print("R2-score: %.2f" % r2_score(y_test_, y_test))



# plt.scatter(X_test, y_test, color='blue')
# plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()








cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X_train.reshape(-1,1), y_train)

y_test_ = regr.predict(X_test.reshape(-1,1))

print("MSE: %.2f" % mean_squared_error(y_test_, y_test))

