import pandas as pd
from sklearn.cluster import KMeans
import joblib


# reading input data
data = pd.read_csv('clustering_input_reduced.csv')
print(data.shape)

'''
# fitting multiple k-means and storing the inertia values (accuracy index)
inertia_list = []
for cluster in range(1,201):    # (this value should be <= no. of data points)
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data)
    inertia_list.append(kmeans.inertia_)
    print(cluster, 'clusters done')

for iner in range(len(inertia_list)):
    result.write('Inertia value using '+str(iner+1)+' clusters = '+str(inertia_list[iner])+'\n')
'''

# specify the no. of clusters
n_clusters = 60

kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
kmeans.fit(data)
pred = kmeans.predict(data)

print(kmeans.inertia_)
print(kmeans.labels_)

frame = pd.DataFrame(data)
frame['cluster'] = pred

# save the clustering result & kmeans model
frame.to_csv('clusters_reduced_result.csv', index=False)
joblib.dump(kmeans, 'clusters_reduced_model.jl') 
