from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# # Display the first digit to understand the data
# plt.imshow(X.iloc[0].values.reshape(28, 28), cmap='gray')
# plt.title(f'Label: {y.iloc[0]}')
# plt.show()

#Data processing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Applying PCA
# Fit PCA with the goal of retaining 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f'Original number of features: {X.shape[1]}')
print(f'Reduced number of features: {X_pca.shape[1]}')

# #Explained Variance
# plt.plot(pca.explained_variance_ratio_.cumsum())
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by Number of Components')
# plt.show()

# #Visualising the data is 2d
# pca_2d = PCA(n_components=2)
# X_pca_2d = pca_2d.fit_transform(X_scaled)

# plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y.astype(int), cmap='tab10')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.title('PCA of MNIST Dataset')
# plt.colorbar(label='Digit Label')
# plt.show()

# #Visualising the data is 3d
# pca_3d = PCA(n_components=3)
# X_pca_3d = pca_3d.fit_transform(X_scaled)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y.astype(int), cmap='tab10')
# ax.set_xlabel('First PC')
# ax.set_ylabel('Second PC')
# ax.set_zlabel('Third PC')
# plt.title('3D PCA of MNIST Dataset')
# plt.show()

#Applying PCA in a Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression(max_iter=1000)),
])

pipeline.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification Reportprint(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()