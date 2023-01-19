# mlspectra-melanin
Program for machine learning inverse design of electronic spectra to structure of DHICA-melanin
## Preparing data for machine learning
### Clustering
Clustering of inter-ring dihedral angles can be done by running 
```
python kmeans_clusters.py
```
for each of the DHICA, DKICA and MKICA.
Subsequently, subclustering of OH dihedral angles within each cluster can be performed based on the clustering result by running 
```
python kmeans_subclusters.py
```
### Spectrum binning and averaging
The next steps are (i) to bin the spectra within a given range and (ii) to calculate the mean spectra and mean structures of every subclusters. Step (ii) is required to discard the redundancies in the dataset. It ensures that we take only one entry from each subcluster instead of all similar/like entires. 
This can be done by running 
```
python subclusters_averaging.py
```
It will create two .csv files; one for geometries and one for spectra.
### Screening important clusters
In a given spectral range, instead of taking all the clusters, we can take most important few clusters which have higher intensity values than others. 
```
python screen_imp_clusters.py
```
will print the desired number of important clusters.
### Final dataset generation
Final dataset generation for ML regarding the important clusters, can be done by running 
```
python make_ML_input-output.py
```
This will create two binary files which contain ML input spectra and output geometries.
## Machine learning training and prediction
Final ML training and predictions can be done by running the following command, 
```
python run_KRR-ML.py --minrange 290 --maxrange 300 --Ntrain 10000
```
`--minrange` and `--maxrange` specify the spectral range and `--Ntrain` specifies the training set size.
## Requirements
```
Numpy
Pandas
Scipy
Scikit-learn
Joblib
```
