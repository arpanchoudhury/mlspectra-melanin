import numpy as np
import pandas as pd
import random 
import MLSpectra


# Read geometry & spectra files and specify important cluster numbers
imp_spec_red, imp_geom_red = MLSpectra.read_data('geom_avg_reduced.csv', 'spectra_100bins_290-300nm_reduced.dat', [59, 52, 39, 31, 25])
imp_spec_mki1, imp_geom_mki1 = MLSpectra.read_data('geom_avg_mki1.csv', 'spectra_100bins_290-300nm_mki1.dat', [26, 40, 24, 41, 52])
imp_spec_dki1, imp_geom_dki1 = MLSpectra.read_data('geom_avg_dki1.csv', 'spectra_100bins_290-300nm_dki1.dat', [3, 20, 15, 29, 30])



X = []
Y = []

res = [[0]*15] # 5red+5mki1+5dki1

NData = 100000
iData = 0
while iData < NData:
    idx = []
    for i in range(len(imp_clusters)):
        idx.append((random.sample(range(imp_spec[i].shape[0]),1))[0])
    for i in range(len(imp_clusters)):
        idx.append((random.sample(range(imp_spec_mki1[i].shape[0]),1))[0])
    for i in range(len(imp_clusters)):
        idx.append((random.sample(range(imp_spec_dki1[i].shape[0]),1))[0])
    if idx not in res:
        res.append(idx)
        iData += 1
        print(iData, 'samples done') 
######

for idx in range(NData):
    Int_sum = np.zeros((spec.shape[1])) 
    alpha = []
    theta = []
    for mol in range(0,5):      # reduced
          Int_sum = Int_sum + imp_spec[mol][res[idx][mol]]
          #alpha.append(list(imp_geom[mol][res[idx][mol]]))
          alpha.append(list(imp_geom[mol][res[idx][mol]])[0:3])
          theta.append(list(imp_geom[mol][res[idx][mol]])[3:])
    
    
    for mol in range(5,10):     # mki1
        Int_sum = Int_sum + imp_spec_mki1[mol-5][res[idx][mol]]
        #alpha.append(list(imp_geom_mki1[mol][res[idx][mol]]))
        alpha.append(list(imp_geom_mki1[mol-5][res[idx][mol]])[0:3])
        theta.append(list(imp_geom_mki1[mol-5][res[idx][mol]])[3:])

    for mol in range(10,15):    # dki1
        Int_sum = Int_sum + imp_spec_dki1[mol-10][res[idx][mol]]
        #alpha.append(list(imp_geom_dki1[mol][res[idx][mol]]))
        alpha.append(list(imp_geom_dki1[mol-10][res[idx][mol]])[0:3])
        theta.append(list(imp_geom_dki1[mol-10][res[idx][mol]])[3:])


    #X.append(Int_sum)
    X.append(Int_sum/len(res[0]))   # average spectra
    flat_alpha = [i for l in alpha for i in l]
    flat_theta = [i for l in theta for i in l]
    flat_Y = list(flat_alpha) + list(flat_theta)
    Y.append(flat_Y)
    print(idx, 'samples done')

X = np.array(X)
Y = np.array(Y)

np.savetxt('X_data_100bins_290-300nm_5Red5MKI5DKI.dat', X)
np.savetxt('Y_data_290-300nm_5Red5MKI5DKI.dat', Y)

