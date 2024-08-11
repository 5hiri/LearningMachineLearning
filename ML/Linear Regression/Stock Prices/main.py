import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

os.chdir(os.path.dirname(__file__))

data = pd.read_csv('indexData.csv')

#Summary statistics
#print(data.describe())

#Display data types of each column
#print(data.dtypes)

#Removing date column and converting categorical columns
data = data.drop('Date', axis=1)
data = pd.get_dummies(data, drop_first=True)

#Create shifted columns(for next day data prediciton)
data['Prev_open'] = data['Open'].shift(1)
data['Prev_high'] = data['High'].shift(1)
data['Prev_low'] = data['Low'].shift(1)
data['Prev_close'] = data['Close'].shift(1)

data['Next_open'] = data['Open'].shift(-1)
data['Next_high'] = data['High'].shift(-1)
data['Next_low'] = data['Low'].shift(-1)
data['Next_close'] = data['Close'].shift(-1)

# #Correlation matrix
# corr_matrix = data.corr()
# print(corr_matrix)

# #create a heatmap using seaborn
# plt.figure(figsize=(18,18))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# #save the heatmap as png
# plt.savefig('correlation_matrix.png')
# #show the plot(optional)
# plt.show()

#check null values:
#print(data.isnull().sum()) 
#backward fill method(use next valid observation to fill gap)
data[['Prev_close', 'Prev_high', 'Prev_low', 'Prev_open', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = data[['Prev_close', 'Prev_high', 'Prev_low', 'Prev_open', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].bfill()
#forward fill method(use previous valid observation to fill gap)
data[['Next_close', 'Next_high', 'Next_low', 'Next_open']] = data[['Next_close', 'Next_high', 'Next_low', 'Next_open']].ffill()
#print(data.isnull().sum()) 

#Feature Selection
y = data[['Next_open', 'Next_high', 'Next_low', 'Next_close']]
X = data.drop(y, axis=1)
X = X.astype(float)

# #checking for multicollinearity
# # Calculate VIF for each feature
# vif_data = pd.DataFrame()
# vif_data["feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# print(vif_data)

#Feature scaling to help stabilize the model and improve performance if performance is bad.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model training
# model = LinearRegression()
#Ridge Method
model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

#Model Evaluation
#predictions
y_pred = model.predict(X_test)
#evaluation
mse = mean_squared_error(y_test, y_pred) #A high MSE value suggests that the model's predictions are significantly off from the actual values. Lower = better.
r2 = r2_score(y_test, y_pred) #For example R-squared = 0.64, this indicates that approximates 64% of the variance in the housing prices is explained by the model. basically higher percentage means better model.

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#cross validation
cross_val_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f'Cross-validated R-squared: {np.mean(cross_val_scores)}')

#coefficients
# Assuming multiple outputs, you might need to handle them separately
for i in range(model.coef_.shape[0]):
    coef_df = pd.DataFrame(model.coef_[i], X.columns, columns=[f'Coefficients_Output_{i}'])
    print(coef_df)

#Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()