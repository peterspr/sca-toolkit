import math


class Welford:
    """
    Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
    http://www.johndcook.com/standard_deviation.html

    Attributes:
        mean (float): current mean
        std_dev (float): current standard deviation
        variance (float): current variance
        n (int): current population size

    Usage:
        welford = Welford()
        list = [1, 2, 3, 4]
        for x in list:
            welford.push(x)
        print(welford.mean)  # 2.5
        print(welford.n)  # 4
        welford.push(5)
        print(welford.variance)  # 2
        print(welford.n)  # 5

    """

    def __init__(self):
        self.n = 0
        self.mean = 0
        self._S = 0

    def push(self, x):
        """Updates the running calculation with a single number."""
        if x is None:
            return
        self.n += 1
        # Performance-wise, the below division is not ideal
        new_mean = self.mean + (x - self.mean) * 1.0 / self.n
        newS = self._S + (x - self.mean) * (x - new_mean)
        self.mean, self._S = new_mean, newS

    @property
    def variance(self):  # This might benefit from caching
        if self.n == 0:
            return 0
        return self._S / self.n  # calculation for population
        # return self._s / (self.n - 1)  # calculation for sample

    @property
    def std_dev(self):
        return math.sqrt(self.variance)

    def __repr__(self):
        return f"<Welford: {self.mean} +- {self.std_dev} var: {self.variance}>"
