from cluster import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
from cluster import KMeans
from cluster import Silhouette
from sklearn.metrics import silhouette_score, silhouette_samples
import pytest
import numpy as np

def test_silhouette():
    """
    Compare Silhouette.score to sklearn.metrics.silhouette_score
    """
    clusters, labels = make_clusters(k=5)
    scoring = Silhouette()
    s_scores = scoring.score(clusters, labels)
    # sklearn.metrics.silhouette_score returns the mean silhouette score
    # sklearn.metrics.silhouette_samples returns the per-sample score
    # Both silhouette_score, silhouette_samples are tested here
    assert np.allclose(s_scores, silhouette_samples(clusters, labels))
    assert np.round(np.mean(s_scores), 2) == np.round(silhouette_score(clusters, labels), 2)
    
    
def test_silhouette_edge_cases():
    """
    Unit tests for Silhouette edge cases
    - Unequal number of observations and number of labels should raise a ValueError
    - Data that is not formatted as a 2D matrix should raise a ValueError
    """
    # not 2D matrix
    with pytest.raises(ValueError) as excinfo:
        scoring = Silhouette()
        scoring.score(np.array([1,2,3]), np.array([1,2,3]))
    # Unequal number of observations and number of labels
    with pytest.raises(ValueError) as excinfo:
        scoring = Silhouette()
        scoring.score(np.array([[1,2],[3,4]]), np.array([1,2,3]))
