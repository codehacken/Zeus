"""
Module to perform clustering of Neural Network layers to understand them.
"""
__author__ = 'ashwin'

import pickle
import numpy as np

# Use Scikit-learn.
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Use Bokeh to plot
from bokeh.plotting import figure
from bokeh.palettes import Spectral10

# Class NeuralKmeans clusters the points in a layer.
class NeuralKMeans():
    def __init__(self, layer):
        self._raw_layer = layer
        self._n_clusters = None
        self._n_centroids = None
        self._layer_tsne = None
        self.layer_tsne_op = None
    
    # Flatten the layer to a single matrix.
    def flatten(self):
        # Flatten the layer matrices to create a single matrix of size of input.
        # Create an empty array of one row.
        self.layer = np.empty([1, np.prod(self._raw_layer[0].shape)])
        for vector in self._raw_layer:
            # Stack the new vectors in array.
            self.layer = np.vstack((self.layer, vector.flatten()))

        # Delete the first row.
        self.layer = np.delete(self.layer, 0, 0)
     
    def cluster(self, num_of_clusters):
        # The total number digits is 10.
        k_means = KMeans(n_clusters=num_of_clusters)
        self._n_clusters = k_means.fit_predict(self.layer)
        self._n_centroids = k_means.cluster_centers_
    
    def tsne(self, n_com=2, random_st=0):
        # Run T-SNE for high dimensional clustering.
        # Optimizations for TSNE.
        # 1. Learning Rate.
        # 2. Angle defining how close the clusters points.
        self._layer_tsne = TSNE(n_components=n_com, random_state=random_st)
        self.layer_tsne_op = self._layer_tsne.fit_transform(self.layer)

    def figure(self, fig_title, n_clusters=None):
        # Once the T-SNE is processed, assign a cluster-ID using K-Means
        # to each data point.
        n_cluster_color = []
        if(n_clusters == None):
            n_clusters = self._n_clusters
        
        for i in n_clusters:
            n_cluster_color.append(Spectral10[i])
                
        b_clusters = figure(title=fig_title)
        b_clusters.scatter(self.layer_tsne_op[:,0], self.layer_tsne_op[:,1],
                           color=n_cluster_color,
                           nonselection_fill_color="#FFFF00", nonselection_fill_alpha=1)
        return b_clusters

        
        