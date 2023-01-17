import os
import argparse as ap
import numpy as np
from sklearn.model_selection import train_test_split
import MLSpectra



# ================= Data loader =================
Dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data/')

parser = ap.ArgumentParser()
parser.add_argument('--minrange', type=str, default='290')
parser.add_argument('--maxrange', type=str, default='300')
parser.add_argument('--Ntrain', type=int, default=10000)
args = parser.parse_args()

min_spec = args.minrange	# spectral range minimum
max_spec = args.maxrange	# spectral range maximum
N_train = args.Ntrain		# training set size

filename = ''.join(i for i in [min_spec,'-',max_spec,'nm'])
print('Specified spectral range:', filename)
print('Size of training set:', N_train)

X = np.load(Dir+'spec_'+filename+'.npy')
Y = np.load(Dir+'geom_'+filename+'.npy')


# ================= Train-Test split =================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=10000)

np.save(filename+'_X_test.npy', X_test)
np.save(filename+'_X_train.npy', X_train)
np.save(filename+'_y_test.npy', Y_test)

indices, indices_q = MLSpectra.gen_index(Dir, X_train, X_test, shuffle=False)


# ================= Kernel specific inputs =================
kernel = 'laplacian'
load_K = False
file_kernel = 'kernel.npy'
lamd = 1e-4
opt_sigma = MLSpectra.single_kernel_sigma(500, X_train, indices, kernel, 'max')
with open(filename+'_opt_sigma.dat', 'w+') as f_sigma:
    f_sigma.write(str(opt_sigma))


# ================= Training =================
K, P = MLSpectra.prepare_trainingdata(kernel,N_train,load_K,file_kernel,indices,lamd,X_train,Y_train,opt_sigma) 

print('Solving matrix equation...')
alpha = MLSpectra.linalg_solve(K,P)
np.save(filename+'_alpha.npy', alpha)

"""
# ================= Prediction =================
print('Predicting...')
out_of_sample_mae = []

for iquery in range(X_test.shape[0]):
    y_pred, _ = MLSpectra.predict(kernel,X_train,X_test,alpha,indices,indices_q,iquery,opt_sigma)
    y_act = Y_test[indices_q[iquery],:]
    phi = np.abs(y_pred - y_act)
    out_of_sample_mae.append(phi)

MAE = np.mean(out_of_sample_mae, axis=0)
np.savetxt(filename+'_MAE.dat', MAE)
"""
print('Done.')
