import os
import numpy as np
import pandas as pd
import MLSpectra


dataDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'clusters/')

# Binning of spectrum 
spec_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/mki1/CAMB3LYP_6-31Gd_spectra')

wavelength_min = 200
wavelength_max = 800

N_bin = 60
read_spec = False            # If true, binned spectra will be loaded from 'file_spec', if false will be calculated and stored in this file
file_spec = 'mki1.dat'
Int_lam, lambda_min, dlambda = MLSpectra.bin_spectra_uniform(spec_path, read_spec, file_spec, wavelength_min, wavelength_max, N_bin)


# Cluster-wise averaging 
df = pd.read_csv(dataDir+'subclusters_mki1_result.csv', index_col=[0])
n_cluster = len(df['cluster'].value_counts().index)

whole_spec = []
for clus_no in range(n_cluster):
    clus_idx = df.index[df['cluster'] == clus_no].tolist()
    clus_df = df.loc[clus_idx,:].drop(columns=['cluster'])
    clus_spec_avg = []

    n_subcluster = len(clus_df['subcluster'].value_counts().index)
    for subclus_no in range(n_subcluster):
        subclus_idx = clus_df.index[clus_df['subcluster'] == subclus_no].tolist()

        tmp_spec = Int_lam[subclus_idx,:]

        clus_spec_avg.append(np.mean(tmp_spec, axis=0))

    clus_spec_avg = np.array(clus_spec_avg)
    whole_spec.append(np.mean(clus_spec_avg, axis=0))
whole_spec = np.array(whole_spec)


# Brute-force search of higher intensity clusters in a wavelength range
N_imp_clusters = 5  # 'N_imp_clusters' most important clusters to be used in training
wave_min = 560      # search will be within wave_min -> (wave_min+resolution) range

resolution = (wavelength_max-wavelength_min)/N_bin
spec = whole_spec[:,int((wave_min-200)/resolution)]
idx_sort = np.argsort(spec)[::-1]

print(N_imp_clusters, 'no of important clusters within', wave_min, 'and', wave_min+resolution, ':')
print(idx_sort[:N_imp_clusters])
