from scipy.spatial.distance import pdist, squareform
import numpy as np

def stochastic_triplet_embedding(triplets, n_dims=2, n_iters=100, alpha=0.1):
    """Perform stochastic triplet embedding on the given triplets.

    Parameters:
    -----------
    triplets : ndarray of shape (n_triplets, 3)
        An array of triplets, where each row contains the indices of the
        triplets.
    n_dims : int (default: 2)
        The number of dimensions to embed the triplets into.
    n_iters : int (default: 100)
        The number of iterations to run the algorithm for.
    alpha : float (default: 0.1)
        The learning rate for the algorithm.

    Returns:
    --------
    ndarray of shape (n_samples, n_dims)
        The embedding of the triplets in the `n_dims` dimensional space.
    """
    # Compute the pairwise distances between all the points
    n_samples = len(triplets)
    distances = squareform(pdist(np.arange(n_samples).reshape(-1, 1)))

    # Initialize the embedding randomly
    embedding = np.random.normal(size=(n_samples, n_dims))

    # Iterate over the triplets and update the embedding using the
    # stochastic gradient descent algorithm
    for i in range(n_iters):
        for triplet in triplets:
            i, j, k = triplet
            d_ij = np.sum((embedding[i] - embedding[j])**2)
            d_ik = np.sum((embedding[i] - embedding[k])**2)
            if d_ij < d_ik:
                gradient = 2 * (embedding[i] - embedding[j] + embedding[k])
                embedding[i] -= alpha * gradient
                embedding[j] += alpha * gradient
                embedding[k] += alpha * gradient
            else:
                gradient = 2 * (embedding[i] - embedding[k] + embedding[j])
                embedding[i] -= alpha * gradient
                embedding[j] += alpha * gradient
                embedding[k] += alpha * gradient
    return embedding
