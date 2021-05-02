# -*- coding: utf-8 -*-
# """
# ===================================
# Demo of DBSCAN clustering algorithm
# ===================================

# Finds core samples of high density and expands clusters from them.

# """

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)

# X = StandardScaler().fit_transform(X)

inputFile = './urbanGB22.txt' # Change this to the data set you want to use.

Y = []
X = []
with open(inputFile, mode='r') as file:
  fileContent = file.read().strip(' ').split("\n")
  # print(fileContent)
  for ind, a in enumerate(fileContent):
  	temp = fileContent[ind].split(',')
  	fileContent[ind] = [float(temp[0]),float(temp[1])]

  # fileContent = file.readlines()
#   n = 2 # number of dimensions
#   Y = [fileContent[i * n:(i + 1) * n] for i in range((len(fileContent) + n - 1) // n )]
# print(Y)
Y = fileContent

for i in range(len(Y)):
  temp = []
  for k in Y[i]:
    temp.append(float(k))
  X.append(temp)


#############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.01, min_samples=20, algorithm='kd_tree').fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# print(labels)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)



