import os
import numpy as np
import pandas as pd
import random 
import mlspectra


dataDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'clusters/')
resDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data/')


# Specify molecular species and their important clusters 
molecules = ['reduced',
             'mki1',
             'dki1']

imp_cluster = [[59, 52, 39, 31, 25],    # reduced clusters
               [26, 40, 24, 41, 52],    # mki cluster 
               [3, 20, 15, 29, 30]]     # dki clusters


# Read geometry & spectra files
imp_spec = []
imp_geom = []
for imol in range(len(molecules)):
    name = molecules[imol]
    spec, geom = mlspectra.read_data(dataDir+'geom_avg_'+name+'.csv', dataDir+'spectra_100bins_290-300nm_'+name+'.dat', imp_cluster[imol])
    imp_spec.append(spec)
    imp_geom.append(geom)


# Make final ML dataset taking random combinations between reduced, mki, dki
# to create the summed spectra
tot_len = sum([len(i) for i in imp_cluster])
idx_list = [[0]*tot_len] 

X = []
Y = []
NData = 20000      # total dataset size
iData = 0
while iData < NData:
    idx = []
    for mol in range(len(imp_cluster)):
        for i in range(len(imp_cluster[mol])):
            idx.append((random.sample(range(imp_spec[mol][i].shape[0]),1))[0])

    if idx not in idx_list:
        idx_list.append(idx)
        iData += 1
        print(iData, 'samples done') 

for iData in range(NData):
    Int_sum = np.zeros((spec[0].shape[1])) 
    theta = []
    phi = []
    j = 0
    for mol in range(len(molecules)):
        for i in range(j, j+len(imp_cluster[mol])):
            idx = idx_list[iData][i]
        
            Int_sum = Int_sum + imp_spec[mol][i-j][idx]
            theta.append(list(imp_geom[mol][i-j][ idx ])[0:3])
            phi.append(list(imp_geom[mol][i-j][ idx ])[3:])

        j += len(imp_cluster[mol])
    
    
    X.append(Int_sum/tot_len)   # average spectra
    flat_theta = [i for l in theta for i in l]
    flat_phi = [i for l in phi for i in l]
    flat_Y = list(flat_theta) + list(flat_phi)
    Y.append(flat_Y)

X = np.array(X)
Y = np.array(Y)

np.save(resDir+'spec_290-300nm.npy', X)
np.save(resDir+'geom_290-300nm.npy', Y)

