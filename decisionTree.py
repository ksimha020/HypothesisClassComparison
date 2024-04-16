#designed by Kshitij Simha R (2022A7PS0572G)
# Importing all necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#important variable declarations (hyperparameters)
look_back = 100

#Random forest Hyperparameters
n_estimators_rf = 100
max_depth_rf = 5
min_samples_split_rf = 2
min_samples_leaf_rf = 1

#Gradient Boosted Tree Hyperparameters
n_estimators_gb = 100
learning_rate_gb = 0.1
max_depth_gb = 3
min_samples_split_gb = 2
min_samples_leaf_gb = 1

#Decision Tree Regressor Hyperparameters
max_depth_dtr = 5
min_samples_split_dtr = 2
min_samples_leaf_dtr = 1

df = pd.read_csv("Data/AMZN.csv")
data = np.array(pd.DataFrame(df['Close']))

x_data, y_data = [], []
for i in range(look_back, len(data)):
    x_data.append(data[i - look_back:i, 0])
    y_data.append(data[i,0])
x_data, y_data = np.array(x_data), np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2,)

tree = DecisionTreeRegressor(max_depth=max_depth_dtr, min_samples_split=min_samples_split_dtr, min_samples_leaf=min_samples_leaf_dtr).fit(x_train, y_train)
forest = RandomForestRegressor(n_estimators=n_estimators_rf, max_depth=max_depth_rf, min_samples_split=min_samples_split_rf, min_samples_leaf=min_samples_leaf_rf).fit(x_train, y_train)
boosted_tree = GradientBoostingRegressor(n_estimators=n_estimators_gb, learning_rate=learning_rate_gb, max_depth=max_depth_gb,  min_samples_split=min_samples_split_gb, min_samples_leaf=min_samples_leaf_gb).fit(x_train, y_train)

y_test_pred_tree = tree.predict(x_test)
y_test_pred_forest = forest.predict(x_test)
y_test_pred_boosted = boosted_tree.predict(x_test)

test_mse = mean_squared_error(y_test, y_test_pred_tree)
print("Test MSE for Decision Tree Regressor:", test_mse)
plt.figure()
plt.title('Decision Tree Regressor vs Test')
plt.plot(y_test_pred_tree, label='predicted')
plt.plot(y_test, label = 'true')
plt.legend()
plt.show()

test_mse = mean_squared_error(y_test, y_test_pred_forest)
print("Test MSE for Random Forest Regressor:", test_mse)
plt.figure()
plt.title('Random Forest Regressor vs Test')
plt.plot(y_test_pred_tree, label='predicted')
plt.plot(y_test, label = 'true')
plt.legend()
plt.show()

test_mse = mean_squared_error(y_test, y_test_pred_boosted)
print("Test MSE for Gradient Boosted Tree:", test_mse)
plt.figure()
plt.title('Gradient Boosted Tree vs Test')
plt.plot(y_test_pred_forest, label='predicted')
plt.plot(y_test, label = 'true')
plt.legend()
plt.show()