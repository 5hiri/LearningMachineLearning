from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

#convert labels to integers
y = y.astype(int)

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Data preprocessing

#normilization: Normalize the pixel values(0-255) to the range [0, 1] to improve performance of the KNN algorithm.
X_train = X_train / 255.0
X_test = X_test / 255.0

#Hyper parameter tuning
# for k in range(1, 22):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     accuracy = knn.score(X_test, y_test)
#     scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
#     print(f'k={k}, Accuracy={accuracy}')
#     print(f'k={k}, Cross-validated accuracy: {scores.mean()}')

#Initialize KNN with a default K value (e.g., 3)

#pca: dimensionality reduction
pca = PCA(n_components=50)  # Reduce to 50 components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)

#train the KNN model
knn.fit(X_train_pca, y_train)

#Model Evaluation
#predict on the test set
y_pred = knn.predict(X_test_pca)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#Classification report
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Visualise misclassification
misclassified_indices = np.where(y_pred != y_test)[0]
for i in misclassified_indices[:10]:
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray') if isinstance(X_test, pd.DataFrame) else plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i]}, Predicted: {y_pred[i]}')
    plt.show()