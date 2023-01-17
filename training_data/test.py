import numpy as np

X = np.loadtxt('X_data_100bins_560-570nm_5MKI5DKI.dat')
Y = np.loadtxt('Y_data_alpha-gamma_560-570nm_5MKI5DKI.dat')

print(X.shape)
print(Y.shape)

np.save('X_data_100bins_560-570nm_5MKI5DKI.npy', X)
np.save('Y_data_alpha-gamma_560-570nm_5MKI5DKI.npy', Y)
