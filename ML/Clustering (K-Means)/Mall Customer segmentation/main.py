import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


os.chdir(os.path.dirname(__file__))

data = pd.read_csv('Mall_Customers.csv')

#print(data.head())

#remove missing data
data = data.dropna()
#Category Processing
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

#Feature Seection
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# #Applying K-Means Clustering
# #Finding Optimal number of clusters (3 or 4)
# inertia = []
# for n in range(1, 11):
#     kmeans = KMeans(n_clusters=n, random_state=42)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)

# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()

#Fit the KMeans Model
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

#Evaluate Clustering
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg}')

#Visualise the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

#Interpret the clusters
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)