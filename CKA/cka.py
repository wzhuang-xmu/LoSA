import numpy as np
import torch
import sys

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel using PyTorch.

    Args:
        x: A num_examples x num_features matrix of features as a PyTorch tensor.

    Returns:
        A num_examples x num_examples Gram matrix of examples as a PyTorch tensor.
    """
    return torch.mm(x, x.t())

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix using PyTorch.

    Args:
        gram: A num_examples x num_examples symmetric matrix as a PyTorch tensor.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
                  estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not torch.allclose(gram, gram.t(), atol=1e-8):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.clone()
    n = gram.size(0)

    if unbiased:
        torch.fill_diagonal_(gram, 0)
        means = torch.sum(gram, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram = gram - means.unsqueeze(0) - means.unsqueeze(1)
        torch.fill_diagonal_(gram, 0)
    else:
        means = torch.mean(gram, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram = gram - means.unsqueeze(0) - means.unsqueeze(1)

    return gram

def cka(gram_x, gram_y, debiased=False):
    """Compute CKA using PyTorch.

    Args:
        gram_x: A num_examples x num_examples Gram matrix as a PyTorch tensor.
        gram_y: A num_examples x num_examples Gram matrix as a PyTorch tensor.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = (gram_x.view(-1) * gram_y.view(-1)).sum()

    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)



def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)