import os
import numpy as np
import pandas as pd
import MLSpectra



dataDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'clusters/')

# ================= Binning of spectrum ================= 
spec_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/reduced/CAMB3LYP_6-31Gd_spectra') 
# spec_path contains csv files with wavelength (nm) and oscillator strength named as 000001.csv and so on

wavelength_min = 560
wavelength_max = 570

N_bin = 100
read_spec = False            # If true, binned spectra will be loaded from 'file_spec', if false will be calculated and stored in this file
file_spec = 'dki1.dat'    
Int_lam, lambda_min, dlambda = MLSpectra.bin_spectra_uniform(spec_path, read_spec, file_spec, wavelength_min, wavelength_max, N_bin)



# ================= Averaging within sub-clusters =================
df = pd.read_csv(dataDir+'subclusters_reduced_result.csv', index_col=[0])
n_cluster = len(df['cluster'].value_counts().index)

geom_avg = []
spec_avg = []
cluster_array = []
for clus_no in range(n_cluster):
    clus_idx = df.index[df['cluster'] == clus_no].tolist()
    clus_df = df.loc[clus_idx,:].drop(columns=['cluster'])

    n_subcluster = len(clus_df['subcluster'].value_counts().index)
    for subclus_no in range(n_subcluster):
        subclus_idx = clus_df.index[clus_df['subcluster'] == subclus_no].tolist()
        subclus_df = clus_df.loc[subclus_idx,:].drop(columns=['subcluster'])

        tmp_geom = subclus_df.to_numpy()
        tmp_spec = Int_lam[subclus_idx,:]

        geom_avg.append(np.mean(tmp_geom, axis=0))
        spec_avg.append(np.mean(tmp_spec, axis=0))
        cluster_array.append(clus_no)

cluster_array = np.array(cluster_array).reshape(-1,1)
geom_avg = np.array(geom_avg)

columns1 = ['theta'+str(i) for i in range(3)]
columns2 = ['phi'+str(i) for i in range(geom_avg.shape[1]-3)]
columns3 = ['cluster']

geom_avg = np.concatenate((geom_avg, cluster_array), axis=1)
geom_df = pd.DataFrame(geom_avg, columns = columns1+columns2+columns3)
spec_avg = np.array(spec_avg)

geom_df.to_csv(dataDir+'geom_avg_dki1.csv', index=False)
#np.savetxt('geom_avg_dki1.csv', geom_avg, delimiter=',')
np.savetxt(dataDir+'spectra_100bins_560-570nm_dki1.dat', spec_avg)


