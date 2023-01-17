import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib




# Features to be clustered in sub-clustering
sub_clus_feat = 'OH_dihedrals_reduced.csv'
data = pd.read_csv(sub_clus_feat)
arr = data.to_numpy()
for j in range(arr.shape[1]):
    for i in range(arr.shape[0]):
        if (abs(arr[i,j]) > 90.0 and arr[i,j] < 0):
            arr[i,j] = arr[i,j] + 360.0


# Data from parent clustering
n_cluster = 60
parent_clus_result = 'clusters_reduced_result.csv'
df = pd.read_csv(parent_clus_result)


mod_list = []
final_df = pd.DataFrame()
for clus_no in range(n_cluster):
    idx = df.index[df['cluster'] == clus_no].tolist()
    data_new = data.iloc[idx,:]

    temp = pd.concat([df.iloc[idx,:], data.iloc[idx,:]], axis=1)
    temp['cluster'] = np.array([clus_no]*len(idx)) 


    # Fitting multiple k-means & choosing right number of cluster using WCSD (within cluster std. dev.)
    iner_list = []
    k_list = list(range(2,len(idx)+1))
    k = 2
    wcsd = 40.0
    while (k in k_list) and (wcsd >= 40.0):     # WCSD threshold is given 40 degree
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(data_new)
        pred = kmeans.predict(data_new)
        iner_list.append(kmeans.inertia_)

        mu = []
        sigma = []
        for i in range(k):
            idx2 = data_new.index[pred == i].tolist()
            mu.append((data.iloc[idx2,:]).mean(axis=0))
            sigma.append((data.iloc[idx2,:]).std(axis=0, ddof=1))

        sigma_max = [sigma[i].max() for i in range(len(sigma))]
        sigma_max = [0 if np.isnan(x) else x for x in sigma_max]

        wcsd = max(sigma_max)
        k = k + 1
        
    opt_k = k - 1

    # Finally cluster according to opt_k
    opt_kmeans = KMeans(n_clusters=opt_k, init='k-means++')
    opt_kmeans.fit(data_new)
    pred = opt_kmeans.predict(data_new)
    tmp['subcluster'] = pred
    #print(frame['cluster'].value_counts())
    mod_list.append(opt_kmeans)
    final_df = pd.concat([final_df, tmp], axis=0)
    print('cluster {} done.'.format(clus_no))



# Save the sub-cluster result
final_df.to_csv('subclusters_reduced_result.csv', index=True, index_label='Mol. index')
joblib.dump(mod_list, 'subclusters_reduced_model.jl')


