import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

data = pd.read_csv('email.csv')

#Summary statistics
#print(data.describe())

#Display data types of each column
#print(data.dtypes)

# Check for missing values: 0 null values
#print(data.isnull().sum()) 

#Data Processing
#Category Processing
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})
# Check for missing values
#print(data.isnull().sum()) #1 null value
#remove null values
data = data.dropna()

#Text Processing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Message'])
y = data['Category']


#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.toarray()) # Convert sparse matrix to array if using TF-IDF

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#Model Training
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

#Model Evaluation
y_pred = model.predict(X_test)

#accuracy
accuracy =  accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#classification report
print(classification_report(y_test, y_pred))

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Model Interpretation
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

#Model Refinement Best Param = 1.0
# param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
# grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
# grid.fit(X_train, y_train)

# print(f'Best Parameters: {grid.best_params_}')