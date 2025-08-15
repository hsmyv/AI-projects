import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df=pd.read_csv(url)


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]



X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

regressor = linear_model.LinearRegression()

regressor.fit(X_train.reshape(-1, 1), y_train)

print ('Coefficients: ', regressor.coef_[0])
print ('Intercept: ',regressor.intercept_)

plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# plt.show()


y_test_ = regressor.predict(X_test.reshape(-1,1))

print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))





# Exercise: 1. Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.¶

# plt.scatter(X_test, y_test,  color='blue')
# plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
# plt.xlabel('Engine size')
# plt.ylabel('Emission')

# plt.show()



# 2. Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets.
# Use the same random state as previously so you can make an objective comparison to the previous training result.

# secondX = cdf.FUELCONSUMPTION_COMB.to_numpy()

# secondX_train, secondX_test, y_train, y_test = train_test_split(secondX, y, test_size=0.2, random_state=42)

# 3. Train a linear regression model using the training data you created.
# Remember to transform your 1D feature into a 2D array.

# regressor2 = linear_model.LinearRegression()
# regressor2.fit(secondX_train.reshape(-1, 1), y_train)

# 4. Use the model to make test predictions on the fuel consumption testing data.


# y_test2_ = regressor2.predict(secondX_test.reshape(-1, 1))
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test2_))
