import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib


dataDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'clusters/')

# reading input data
data = pd.read_csv(dataDir+'clustering_input_reduced.csv')

# kmeans clustering
n_clusters = 60     # number of clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
kmeans.fit(data)
pred = kmeans.predict(data)
"""print(kmeans.inertia_)
print(kmeans.labels_)"""
df = pd.DataFrame(data)
df['cluster'] = pred

# within cluster standard deviation
fout = open(dataDir+'wcsd.dat', 'w+')
for clus_no in range(n_clusters):
    clus_idx = df.index[df['cluster'] == clus_no].tolist()
    clus_df = df.loc[clus_idx,:]
    fout.write('Cluster = '+str(clus_no)+'\n')
    P = clus_df.to_numpy()
    meanP = np.mean(P, axis=0)
    stdP = np.std(P, axis=0)
    for i in range(3):
        fout.write(str(meanP[i])+' +- '+str(stdP[i])+'\n')
print('Within cluster standard deviation is written in wcsd.dat file.')

# save the clustering result & kmeans model
df.to_csv(dataDir+'clusters_reduced_result.csv', index=False)
joblib.dump(kmeans, dataDir+'clusters_reduced_model.jl') 
