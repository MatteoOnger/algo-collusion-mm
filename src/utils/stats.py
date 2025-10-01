import numpy as np



class OnlineVectorStats:
    """
    Tracks the running mean and standard deviation for each component 
    of input vectors using Welford's algorithm.

    See Also
    --------
    - Welford, B. P. (1962). Note on a method for
    calculating corrected sums of squares and products.
    Technometrics, 4(3), 419-420.
    """

    def __init__(self, dim: int):
        """
        Initialize the statistics tracker.

        Parameters
        ----------
        dim : int
            Dimension of the input vectors.
        """
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.rss = np.zeros(dim, dtype=np.float64)
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

        if x.shape != self.mean.shape:
            raise ValueError(f"Expected shape {self.mean.shape}, got {x.shape}")

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.rss += delta * delta2
        return


    def get_mean(self) -> np.ndarray|None:
        """
        Get the current running mean.

        Returns
        -------
        : np.ndarray or None
            Running mean of shape (dim,), or None if no data has been added yet.
        """
        return self.mean if self.n > 0 else None


    def get_std(self) -> np.ndarray|None:
        """
        Get the current running standard deviation (sample std).

        Returns
        -------
        : np.ndarray or None
            Running standard deviation of shape (dim,), or None if fewer than 2 vectors have been added.
        """
        if self.n < 2:
            return None
        return np.sqrt(self.rss / (self.n - 1))


    def get_count(self) -> int:
        """
        Get the number of vectors processed.

        Returns
        -------
        : int
            Total number of updates.
        """
        return self.n
