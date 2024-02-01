from cluster import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
from cluster import KMeans
from cluster import Silhouette
import pytest
import numpy as np

def test_kmeans():
    """
    Generate different clusters and test fit, predict, get_error, and get_centroid work
    """
    # Test on tight clusters
    t_clusters, t_labels = make_clusters(k=3, scale=0.3)
    model = KMeans(k=3)
    model.fit(t_clusters)
    predictions = model.predict(t_clusters)
    # Clusters can be identical, but cluster labels may not necessarily be the same
    k = 3
    clusters, labels = t_clusters, t_labels
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = set(clusters[np.argwhere(labels==i).flatten()].flatten())
    # Higher-dimensional arrays cannot be converted to sets.
    # Values in clusters are random floats, so I will assume sets of the 
    # flattened matrices are unique enough for the equality test below to be reasonable
    for i in range(k):
        pred_c = set(clusters[np.argwhere(np.array(predictions)==i).flatten()].flatten())
        assert pred_c in cluster_dict.values()
        
    # Test on k=1 clusters
    k1_clusters, k1_labels = make_clusters(k=1, n=10)
    model = KMeans(k=1)
    model.fit(k1_clusters)
    predictions = model.predict(k1_clusters)
    k = 1
    clusters, labels = k1_clusters, k1_labels
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = set(clusters[np.argwhere(labels==i).flatten()].flatten())
    for i in range(k):
        pred_c = set(clusters[np.argwhere(np.array(predictions)==i).flatten()].flatten())
        assert pred_c in cluster_dict.values()
    
    # Check get_centroids
    assert np.allclose([np.mean(k1_clusters, axis=0)], model.get_centroids())
                          
    # Test on high dimensionality data
    highd_clusters, highd_labels = make_clusters(n=100, m=200, k=3)
    model = KMeans(k=3)
    model.fit(highd_clusters)
    predictions = model.predict(highd_clusters)
    k = 3
    clusters, labels = highd_clusters, highd_labels
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = set(clusters[np.argwhere(labels==i).flatten()].flatten())
    for i in range(k):
        pred_c = set(clusters[np.argwhere(np.array(predictions)==i).flatten()].flatten())
        assert pred_c in cluster_dict.values()
                          
    # Test that fitting exits when the maximum number of iterations is met
    model = KMeans(k=2, max_iter=0)
    model.fit(clusters)
    # Error is not updated if fitting exits immediately
    assert model.error == np.inf
    # Test that fitting exits when the error tolerance between iterations is met
    model = KMeans(k=2, tol=np.inf)
    model.fit(clusters)
    assert model.error == np.inf


def test_kmeans_edge_cases():
    """
    Unit tests for KMeans edge cases
    - k = 0 should raise a ValueError
    - k > number of observations in the data should raise a ValueError
    - Data that is not formatted as a 2D matrix should raise a ValueError
    - Predicting on data that is not formatted as a 2D matrix
      and that does not have the same number of features as the data used to fit the model
      should raise a ValueError
    """
    # k = 0
    with pytest.raises(ValueError) as excinfo:
        model = KMeans(k=0)
    # k > number of observations
    with pytest.raises(ValueError) as excinfo:
        clusters, labels = make_clusters(n=100, m=2, k=3)
        model = KMeans(k=10000)
        model.fit(clusters)
    # not 2D matrix
    with pytest.raises(ValueError) as excinfo:
        model = KMeans(k=2)
        model.fit(np.array([1,2,3]))
    # a 2D matrix where observations have 1 feature is okay
    model = KMeans(k=3)
    model.fit(np.array([[1.0],[2.0],[3.0]]))
    assert set(model.get_centroids().flatten()) == {1.0, 2.0, 3.0}
    # must predict on data that is a 2D matrix
    with pytest.raises(ValueError) as excinfo:
        model.predict(np.array([1,2,3]))
    # must predict on data that is has the same number of features
    with pytest.raises(ValueError) as excinfo:
        clusters, labels = make_clusters(n=20, m=2, k=3)
        model = KMeans(k=3)
        model.fit(clusters)
        clusters, labels = make_clusters(n=10, m=5, k=3)
        model.predict(clusters)
