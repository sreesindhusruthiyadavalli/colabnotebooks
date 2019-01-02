import numpy as np
import pandas as pd
from kmodes.kmodes import KModes

# random categorical data
#data = np.random.choice(20, (100, 10))

file_name ='labelled_mean_na.csv'
df = pd.read_csv(file_name, low_memory=False)

#kp = KPrototypes(n_clusters=3, init='Huang', n_init=1, verbose=True)
#kp.fit_predict(data, categorical=[4])

#print(kp.cluster_centroids_)
#print(kp.labels_)

km = KModes(n_clusters=4, init='Huang', n_init=2, verbose=1)

clusters = km.fit(df)
print(clusters)
print(clusters.shape)


# Print the cluster centroids
print(type(km.cluster_centroids_))
print(km.cluster_centroids_.shape)
print(km.cluster_centroids_)
