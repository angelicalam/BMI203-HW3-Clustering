import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k < 1:
            raise ValueError(f"The number of centroids k for cluster fitting must be greater than 0.")
            
        # Assigns observations to clusters. E.g., [1, 1, 2, 0, 2] means that
        # the first and second observations are assigned to cluster 1, and so forth.
        self.clusters = []
        # Holds cluster centroids. Index of centroid value identifies its associated cluster.
        self.centroids = []
        # Holds mean squared error of the fitted model
        self.error = np.inf
        # Save hyperparameters
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if len(mat.shape) != 2:
            raise ValueError(f"Data must be formatted as a 2D matrix.")
        if len(mat) < self.k:
            raise ValueError("K-means clustering cannot be performed when the number "
                            + f"of observations, {mat.shape[0]}, is higher than k={self.k}")
            
        # Initialize cluster centers as described for k-means++
        self.centroids = np.zeros((self.k, mat.shape[1]))
        self.centroids[0] = mat[np.random.randint(mat.shape[0])]
        for center in range(1, self.k):
            # Distance between two observations is the L2 norm of their difference vector
            # Compute distance D(x) between each observation and the nearest already chosen center
            dist_to_center = []
            for i in range(mat.shape[0]):
                # If mat[i] == self.centroids[c], the distance is zero
                # and will have no impact on the weighted probability distribution
                dx = np.linalg.norm([mat[i] - self.centroids[c] for c in range(center)], axis=1).min()
                dist_to_center.append(dx)
            # Turn the distances into a weighted probability distribution
            chosen = np.random.uniform(np.sum(dist_to_center))
            tot_dist, new_center_i = dist_to_center[0], 0
            # Choose a new center with probability proportional to D(x)
            while chosen > tot_dist:
                new_center_i += 1
                tot_dist += dist_to_center[new_center_i]
            self.centroids[center] = mat[new_center_i]
        # Assign observations to their nearest cluster centers.
        # This concludes the k-means++ implementation of the first step of Lloyd's algorithm,
        # where the observations are partitioned into k sets.
        self.clusters = self.update_clusters(mat)

        # Use Lloyd's algorithm to update clusters until convergence, either
        # of cluster assignment or error, or the max number of iterations has been met
        prev_clusters = [None for i in range(len(self.clusters))]
        prev_error = 0
        for i in range(self.max_iter):
            # Break when cluster assignment converges
            if np.array_equal(prev_clusters, self.clusters):
                print(prev_clusters, self.clusters)
                break
            # Break when the difference in error after updating is <= than the tolerance
            if np.abs(prev_error - self.error) <= self.tol:
                break
            prev_clusters = self.clusters
            prev_error = self.error
            # Update centroids
            self.update_centroids(mat)
            # Update cluster assignments
            self.update_clusters(mat)
            # Calculate mean squared error of fitted model
            self.error = 0
            for i in range(self.k):
                center = self.centroids[i]
                idx = np.argwhere(np.array(self.clusters) == i).flatten()
                self.error += np.sum((mat[idx] - center)**2)
            self.error = np.mean(self.error)

        
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if len(mat.shape) != 2:
            raise ValueError(f"Data must be formatted as a 2D matrix.")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Observations in data must have the same number of features "
                            + "as in the data used to fit the model")
            
        # Assign observations in mat to clusters defined by self.clusters,
        # using the metric of shortest Euclidean distance to centroids.
        # This does not actually update self.clusters.
        return self.update_clusters(mat)

    
    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

        
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids

    
    def update_centroids(self, mat):
        """
        Update centroids of dataset mat given cluster assignments self.clusters.
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self.centroids = np.zeros((self.k, mat.shape[1]))
        for i in range(self.k):
            # Get indices of observations in mat that are in cluster i
            idx = np.argwhere(np.array(self.clusters) == i).flatten()
            # Calculate and update the centroid of the cluster
            self.centroids[i] = np.mean(mat[idx], axis=0)
            
    
    def update_clusters(self, mat, centers=None):
        """
        Return updated cluster assignments.
        For each observation x in mat, get the index (i.e., identity) of the cluster 
        whose center is closest in distance to x, as calculated by the L2-norm of the 
        difference vector (i.e., Euclidean distance).
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
            centers: np.ndarray
                A 2D matrix representing the cluster centers. Is self.centroids by default
            clusters: np.ndarray
                A 1D array with the cluster label for each of the observations in `mat`.
                Is self.clusters by default
                
        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if centers == None:
            centers = self.centroids
        clusters = []
        for x in mat:
            # Calculate distances between observation x and each centroid
            distances = np.linalg.norm([x - c for c in centers], axis=1)
            # Assign x to the cluster with the nearest centroid
            clusters.append( np.argmin(distances) )
        return clusters
