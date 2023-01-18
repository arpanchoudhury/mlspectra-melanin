from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import random



def read_files(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = sorted(files)

    return files


def read_data(geomFile, specFile, imp_clusters):
    df = pd.read_csv(geomFile)
    n_cluster = len(df['cluster'].value_counts().index)
    geom = df.iloc[:,:-1].to_numpy()
    spec = np.loadtxt(specFile)

    imp_spec = []
    imp_geom = []
    for cluster in imp_clusters:
        clus_idx = df.index[df['cluster'] == cluster].tolist()
        tmp_spec = spec[clus_idx,:]
        tmp_geom = geom[clus_idx,:]
        imp_spec.append(tmp_spec)
        imp_geom.append(tmp_geom)

    return imp_spec, imp_geom



def gen_index(Dir, X_train, X_test, shuffle):

    if shuffle:
        indices_t = list(random.sample(range(X_train.shape[0]), X_train.shape[0]))
        indices_q = list(random.sample(range(X_test.shape[0]), X_test.shape[0]))
    else:
        indices_t = list(i for i in range(X_train.shape[0]))
        indices_q = list(i for i in range(X_test.shape[0]))

    with open(Dir+'train_index.dat', 'w+') as f_idx_t:
        for i in range(len(indices_t)):
            f_idx_t.write("%06d"%(indices_t[i])+'\n')

    with open(Dir+'test_index.dat', 'w+') as f_idx_q:
        for i in range(len(indices_q)):
            f_idx_q.write("%06d"%(indices_q[i])+'\n')

    return indices_t, indices_q

