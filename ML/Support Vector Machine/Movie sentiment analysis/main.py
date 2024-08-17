import pandas as pd
import os, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

os.chdir(os.path.dirname(__file__))

# data = pd.read_csv('IMDB Dataset.csv')

#display the first few rows
#print(data.head())

# #Data Preprocessing
# nltk.download('stopwords')

# def preprocess_text(text):
#     text = re.sub(r'\W', ' ', text) # Remove punctuation
#     text = text.lower() #Convert to lowercase
#     text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) #Remove single characters
#     text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters at the start
#     text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with a single space
#     text = re.sub(r'^b\s+', '', text)  # Remove prefixed 'b'
#     text = re.sub(r'\s+', ' ', text)  # Remove additional spaces
    
#     ps = PorterStemmer()
#     text = ' '.join([ps.stem(word) for word in text.split() if word not in set(stopwords.words('english'))])

#     return text

# data['review'] = data['review'].apply(preprocess_text)
# data.to_csv('IMDB Dataset Processed.csv', encoding='utf-8', index=False)
# print("Preprocessing complete.")
data = pd.read_csv('IMDB Dataset Processed.csv')
data = data.sample(n=5000, random_state=42)

#Text Vectorization(convert text to numerical data)
vectorizer = TfidfVectorizer(max_features=5000) #Max features can be adjusted
X = vectorizer.fit_transform(data['review']).toarray()
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

#Splitting data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
print("Training Complete.")

#Model Evaluation
y_pred = svm_model.predict(X_test)

#Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# #Hyperparameter Tuning
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear', 'rbf'],
#     'gamma': [1, 0.1, 0.01, 0.001]
# }

# grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
# grid_search.fit(X_train, y_train)

# print(f'Best parameters: {grid_search.best_params_}')

#Model interpretation and Feature Importance
#Only for linear kernel
if svm_model.kernel == 'linear':
    feature_importance = svm_model.coef_.flatten()
    top_indices = feature_importance.argsort()[-10:][::-1]
    print("Top 10 features:")
    for index in top_indices:
        print(vectorizer.get_feature_names_out()[index], feature_importance[index])

#Cross validation
scores = cross_val_score(svm_model, X, y, cv=5)
print(f'Cross-validated accuracy: {scores.mean()}')

#Training the SVM model
svm_model = SVC(C=1, kernel='rbf', gamma=1, random_state=42)
svm_model.fit(X_train, y_train)
print("Hypertuned Model:")

#Model Evaluation
y_pred = svm_model.predict(X_test)

#Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

#Cross validation
scores = cross_val_score(svm_model, X, y, cv=5)
print(f'Cross-validated accuracy: {scores.mean()}')