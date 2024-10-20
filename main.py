import pandas as pd
import numpy as np

df = pd.read_csv('nasabah.csv')

X = df[['Usia', 'Saldo']]
k = 3

initial_centroids = X.sample(n=k, random_state=30).values

def manhattan_kmeans(X, initial_centroids, max_iters=100):
  centroids = initial_centroids
  for i in range(max_iters):
    distances = np.abs(X.values[:, np.newaxis] - centroids).sum(axis=2)
    closest_centroids = np.argmin(distances, axis=1)
    
    new_centroids = np.array([X.values[closest_centroids == j].mean(axis=0) for j in range(k)])
    
    if np.all(centroids == new_centroids):
      break
    centroids = new_centroids
    
  return closest_centroids, centroids

cluster_labels, final_centroids = manhattan_kmeans(X, initial_centroids)

df['Cluster'] = cluster_labels

print(df)
print(final_centroids)
