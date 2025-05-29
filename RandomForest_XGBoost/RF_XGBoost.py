#Comparing Random Forest and XGBoost modeling performance


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Kaliforniya mənzil qiymətləri məlumat dəstini yükləyirik
data = fetch_california_housing()
X, y = data.data, data.target  # X – xüsusiyyətlər, y – hədəf dəyişəni (qiymət)

# Məlumatları telim və test dəstlərinə bölürük (80% telim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mushahidelerin və xüsusiyyətlərin sayını çap edirik
N_observations, N_features = X.shape
print('Mushahidelerin sayı: ' + str(N_observations))
print('Xususiyyetlerin sayı: ' + str(N_features))

# Modeldə istifadə olunacaq ağacların (estimatorların) sayı
n_estimators = 100

# Random Forest (Təsadüfi Meşə) və XGBoost modellərini yaradırıq
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# Modelləri öyrədirik və telim vaxtını ölçürük

# Random Forest üçün telim vaxtının ölçülməsi
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

# Random Forest üçün telim vaxtını çap edirik
print(f"Random Forest telim muddeti: {rf_train_time:.4f} saniyə")

# XGBoost üçün telim vaxtının ölçülməsi
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb

# XGBoost üçün telim vaxtını çap edirik
print(f"XGBoost telim muddeti: {xgb_train_time:.4f} saniyə")













#Exercise 2. Use the fitted models to make predictions on the test set.¶
# Also, measure the time it takes for each model to make its predictions using the time.time() function to measure the times before and after each model prediction.

# Measure prediction time for Random Forest
# start_time_rf = time.time()
# y_pred_rf = rf.predict(X_test)
# end_time_rf = time.time()
# rf_pred_time = end_time_rf - start_time_rf

# # Measure prediciton time for XGBoost
# start_time_xgb = time.time()
# y_pred_xgb = xgb.predict(X_test)
# end_time_xgb = time.time()
# xgb_pred_time = end_time_xgb - start_time_xgb

# print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
# print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')





# # Exercise 3: Calulate the MSE and R^2 values for both models
# y_pred_rf = rf.predict(X_test)
# y_pred_xgb = xgb.predict(X_test)

# mse_rf = mean_squared_error(y_test, y_pred_rf)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)
# r2_rf = r2_score(y_test, y_pred_rf)
# r2_xgb = r2_score(y_test, y_pred_xgb)
# print(f"Random Forest MSE: {mse_rf:.4f}, R^2: {r2_rf:.4f}")
# print(f"XGBoost MSE: {mse_xgb:.4f}, R^2: {r2_xgb:.4f}")

# #You can see from the MSE and R^2 values that XGBoost is better than Random Forest, but the differences aren't overwhelming.









#Exercise 6. Calculate the standard deviation of the test data
# std_y = np.std(y_test)

# start_time_rf = time.time()
# y_pred_rf = rf.predict(X_test)
# end_time_rf = time.time()
# rf_pred_time = end_time_rf - start_time_rf

# # Measure prediciton time for XGBoost
# start_time_xgb = time.time()
# y_pred_xgb = xgb.predict(X_test)
# end_time_xgb = time.time()
# xgb_pred_time = end_time_xgb - start_time_xgb

# mse_rf = mean_squared_error(y_test, y_pred_rf)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)
# r2_rf = r2_score(y_test, y_pred_rf)
# r2_xgb = r2_score(y_test, y_pred_xgb)

# print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
# print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')
# print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
# print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')
# std_y = np.std(y_test)



# plt.figure(figsize=(14, 6))

# # Random Forest plot
# plt.subplot(1, 2, 1)
# plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue",ec='k')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
# plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
# plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
# plt.ylim(0,6)
# plt.title("Random Forest Predictions vs Actual")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.legend()


# # XGBoost plot
# plt.subplot(1, 2, 2)
# plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange",ec='k')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
# plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
# plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
# plt.ylim(0,6)
# plt.title("XGBoost Predictions vs Actual")
# plt.xlabel("Actual Values")
# plt.legend()
# plt.tight_layout()
# plt.show()