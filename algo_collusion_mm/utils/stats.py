""" Utility module for tracking statistics. 
"""
import numpy as np

from typing import Tuple



class OnlineVectorStats:
    """
    Efficiently tracks running statistics (mean, standard deviation, min, max)
    for each component of input vectors using Welford's online algorithm.

    Attributes
    ----------
    dim : int or tuple of ints
        Dimension of the vectors.

    References
    ----------
    - Welford, B. P. (1962). Note on a method for calculating
    corrected sums of squares and products. Technometrics, 4(3), 419-420.
    """

    def __init__(self, dim: int|Tuple[int, ...]):
        """
        Parameters
        ----------
        dim : int or tuple of ints
            Dimension of the input vectors.
        """
        self.dim = dim
        """Vectors dimension."""

        self._n = 0
        """Number of vectors processed so far."""
        self._mean = np.zeros(dim, dtype=np.float64)
        """Running mean of the input vectors."""
        self._rss = np.zeros(dim, dtype=np.float64)
        """Running sum of squares of differences."""
        self._min = np.full(dim, np.inf, dtype=np.float64)
        """Minimum value seen so far."""
        self._max = np.full(dim, -np.inf, dtype=np.float64)
        """Maximum value seen so far."""
        return


    def update(self, x: np.ndarray) -> None:
        """
        Update the running statistics with a new vector.

        Parameters
        ----------
        x : np.ndarray
            New input vector of shape (dim,).

        Raises
        ------
        ValueError
            If the input vector does not match the expected dimension.
        """
        x = np.asarray(x, dtype=np.float64)

        if x.shape != self._mean.shape:
            raise ValueError(f"Expected shape {self._mean.shape}, got {x.shape}")

        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._rss += delta * delta2
    
        self._min = np.minimum(self._min, x)
        self._max = np.maximum(self._max, x)
        return


    def get_mean(self) -> np.ndarray|None:
        """
        Get the current running mean.

        Returns
        -------
        : np.ndarray or None
            Running mean of shape (dim,), or None if no data has been added yet.
        """
        return self._mean if self._n > 0 else None


    def get_std(self,  sample: bool = True) -> np.ndarray|None:
        """
        Get the current running standard deviation.

        Parameters
        ----------
        sample : bool, default=True
            If True, return the sample standard deviation (dividing by n-1).
            If False, return the population standard deviation (dividing by n).

        Returns
        -------
        : np.ndarray or None
            Running standard deviation of shape (dim,), or None if not enough vectors have been added.
        """
        if (sample and self._n < 2) or (not sample and self._n < 1):
            return None
        denom = (self._n - 1) if sample else self._n
        return np.sqrt(self._rss / denom)


    def get_min(self) -> np.ndarray|None:
        """
        Get the minimum value seen so far for each component.

        Returns
        -------
        : np.ndarray or None
            Running minimum of shape (dim,), or None if no data has been added yet.
        """
        return self._min if self._n > 0 else None


    def get_max(self) -> np.ndarray|None:
        """
        Get the maximum value seen so far for each component.

        Returns
        -------
        : np.ndarray or None
            Running maximum of shape (dim,), or None if no data has been added yet.
        """
        return self._max if self._n > 0 else None


    def get_count(self) -> int:
        """
        Get the number of vectors processed.

        Returns
        -------
        : int
            Total number of updates.
        """
        return self._n
