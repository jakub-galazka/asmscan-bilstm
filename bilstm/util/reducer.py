import numpy as np

from umap import UMAP
from sklearn.manifold import TSNE


def tsne_2d(mdim_representation: any) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
        T-distributed Stochastic Neighbor Embedding
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        
        Parameters
        ----------
        mdim_representation : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Multidimensional representation.

        Returns
        -------
        Embedding of the multidimensional representation in 2D space.

        x : np.ndarray[np.float64]
            X dimension.

        y : np.ndarray[np.float64]
            Y dimension.
    '''
    tsne = TSNE(verbose=1).fit_transform(mdim_representation)
    return tsne[:, 0], tsne[:, 1] # x, y

def umap_2d(mdim_representation: any, n_neighbors: int = 5) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
        Uniform Manifold Approximation and Projection
        https://umap-learn.readthedocs.io/en/latest/
        
        Parameters
        ----------
        mdim_representation : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Multidimensional representation.

        n_neighbors : int
            This parameter controls how UMAP balances local versus global structure in the data.
            It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data.
            This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture),
            while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data,
            losing fine detail structure for the sake of getting the broader of the data.

        Returns
        -------
        Embedding of the multidimensional representation in 2D space.

        x : np.ndarray[np.float64]
            X dimension.

        y : np.ndarray[np.float64]
            Y dimension.
    '''
    umap = UMAP(verbose=1, n_neighbors=n_neighbors).fit_transform(mdim_representation)
    return umap[:, 0], umap[:, 1] # x, y
