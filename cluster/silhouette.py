import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if len(X.shape) != 2:
            raise ValueError(f"Data must be formatted as a 2D matrix.")
        if len(X) != len(y):
            raise ValueError("The number of observations and labels must be the same.")
        
        cluster_labels = np.unique(y)
        k = len(cluster_labels)
        # Get mean similarity - within cluster mean distance
        a, b = np.zeros(len(y)), np.zeros(len(y))
        for x in range(len(y)):
            cluster_i = y[x]
            # Get mean similarity - within cluster mean distance
            a[x] = self.mean_similarity(X[x], X[np.argwhere(y==cluster_i).flatten()])
            # Get mean dissimilarity - between cluster mean distance
            b[x] = self.mean_dissimilarity(X[x], X[np.argwhere(y!=cluster_i).flatten()], 
                                                 y[np.argwhere(y!=cluster_i).flatten()])
        # Calculate silhouette score
        return (b - a) / np.array([a,b]).max(axis=0)
        
        
    def mean_similarity(self, x, cluster):
        """
        Return the mean similarity score for observation x.
        This is the mean distance between x and all other observations in the same cluster.

        inputs:
            x: np.ndarray
                A 1D array that is a single observation x.

            cluster: np.ndarray
                a 2D matrix that consists of observations in the same cluster as x

        outputs:
            float
                the mean similarity score for observation x
        """
        # Calculate Euclidean distances between x and each observation in cluster.
        # The distance between x and the same x in cluster is zero, so I do not need
        # to exclude calculating this distance explicitly.
        distances = np.linalg.norm((cluster - x), axis=1)
        return distances.sum() / (len(cluster) - 1)
        
        
    def mean_dissimilarity(self, x, clusters, y):
        """
        Return the mean dissimilarity score for observation x
        This is the mean distance between x and all observations in clusters that are
        not the cluster assigned to x.

        inputs:
            x: np.ndarray
                A 1D array that is a single observation x.

            cluster: np.ndarray
                a 2D matrix that consists of observations in all clusters except
                the cluster assigned to x
                
            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            float
                the mean dissimilarity score for observation x
        """
        closest = np.inf
        for i in np.unique(y):
            # Calculate Euclidean distances between x and each observation in cluster c.
            c = clusters[np.argwhere(y==i).flatten()]
            distances = np.linalg.norm( c - x, axis=1 )
            mean_dist = distances.sum() / len(c)
            # Keep the mean distance between x and observations in the closest cluster
            if mean_dist < closest:
                closest = mean_dist
        return closest
            