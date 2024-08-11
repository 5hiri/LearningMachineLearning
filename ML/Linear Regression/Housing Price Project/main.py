import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from scipy import stats
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA



os.chdir(os.path.dirname(__file__))

scaler = StandardScaler()

#load dataset
data = pd.read_csv('Housing_Price_Data.csv')

#Summary statistics
#print(data.describe())

#Display data types of each column
#print(data.dtypes)

#Converting categorical variables using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

#Correlation matrix
#corr_matrix = data.corr()
#print(corr_matrix)

#create a heatmap using seaborn
#plt.figure(figsize=(18,18))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#save the heatmap as png
#plt.savefig('correlation_matrix.png')
#show the plot(optional)
#plt.show()

#Handling missing values
#fill the missing values(optional)
data = data.fillna(data.mean())
#remove the missing values(optional)
#data = data.dropna()


#Feature Selection
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad_yes', 'guestroom_yes', 'basement_yes', 'airconditioning_yes', 'prefarea_yes', 'furnishingstatus_unfurnished']]
#Outlier Detection and removal that could be skewing the model if performance is bad.
# Convert all columns to float
X = X.astype(float)
X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
y = data['price']
#filtering out the same things that were filtered from X
y = y[X.index]

#checking for multicollinearity
# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

#reducing multicollinearity
#combining variabls
# X['rooms'] = X['bedrooms'] + X['bathrooms']
# X = X.drop(columns=['bedrooms', 'bathrooms'])
#dropping variabls
#X = X.drop(columns=['mainroad_yes'])

#Feature scaling to help stabilize the model and improve performance if performance is bad.
X_scaled = scaler.fit_transform(X)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model training
#initiate model
#model = LinearRegression()
#Ridge regression(Ridge regression penalizes large coefficients, which can help reduce the impact of multicollinearity.)
model = Ridge(alpha=0.001)
#Lasso regression
#model = Lasso(alpha=0.1)
#train the model
model.fit(X_train, y_train)

#Model evaluation
#predictions
y_pred = model.predict(X_test)
#evaluation
mse = mean_squared_error(y_test, y_pred) #A high MSE value suggests that the model's predictions are significantly off from the actual values. Lower = better.
r2 = r2_score(y_test, y_pred) #For example R-squared = 0.64, this indicates that approximates 64% of the variance in the housing prices is explained by the model. basically higher percentage means better model.

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Coefficients
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficients'])
print(coef_df)

#Model Refinement
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f'Cross-validated R-squared: {scores.mean()}')