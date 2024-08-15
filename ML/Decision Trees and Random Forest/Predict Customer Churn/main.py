import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns

os.chdir(os.path.dirname(__file__))

data = pd.read_csv('Telco-Customer-Churn.csv')

#Display first few rows
#print(data.head())

#Check for missing values. 
#print(data.isnull().sum()) #Missing Values = 0

#Encode Categorical Variables
#look for binary categorical columns
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
# for col in data.columns:
#     if data[col].dtype == 'object' or data[col].nunique() == 2:
#         unique_values = data[col].nunique()
#         if unique_values == 2:
#             binary_columns.append(col)
#print("Binary categorical columns:", binary_columns)

#Convert binary categorical columns
for col in binary_columns:
    unique_vals = data[col].unique()
    if len(unique_vals) == 2:
        data[col] = data[col].map({unique_vals[0]: 0, unique_vals[1]: 1})

# One-hot encode other categorical variables
data = pd.get_dummies(data, drop_first=True)

#Feature Selection and split
X = data.drop('Churn', axis=1)
y = data['Churn']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Implementing Decision trees
#Initialize and train the decision tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

#Initialize and train random forest model
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

#Evaluate the model

#Decision Tree Model
# Predict on the test set
y_pred = tree_model.predict(X_test)

#Evaluate the decision tree model
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Model:\nAccuracy: {accuracy}')
print(classification_report(y_test, y_pred) + "\n\n")

# #Plot the decision tree
# plt.figure(figsize=(18,8))
# plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
# plt.show()

#Random Forest Tree
# Predict on the test set
y_pred_forest = forest_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f'Random Forest Model:\nAccuracy: {accuracy_forest}')
print(classification_report(y_test, y_pred_forest))

# # Get feature importances from the Random Forest model
# importances = forest_model.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # Plot the feature importances
# plt.figure(figsize=(10,6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance')
# plt.show()

#Hyper parameter tuning(grid search, for random forest) 
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# grid_search = GridSearchCV(estimator=forest_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)
# print(f'Best parameters: {grid_search.best_params_}')

#random forest model after hypertuning
#Initialize and train random forest model
forest_model = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, min_samples_split=2, max_depth=None, random_state=42)
forest_model.fit(X_train, y_train)

# Predict on the test set
y_pred_forest = forest_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f'Random Forest Model after hypertuning:\nAccuracy: {accuracy_forest}')
print(classification_report(y_test, y_pred_forest))