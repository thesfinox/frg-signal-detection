"""
Distribution test

Test functions and methods of the base distribution class.
"""

import pytest

from frg.distributions.distributions import Distribution


class TestBaseDistribution:
    """Test the base distribution"""

    dist = Distribution()

    def test_ipdf(self):
        """Test the PDF of the inverse distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.ipdf(0.0)

    def test_icdf(self):
        """Test the CDF of the inverse distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.icdf(0.0)

    def test_dipdf(self):
        """Test the derivative of the PDF of the inverse distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.dipdf(0.0)
