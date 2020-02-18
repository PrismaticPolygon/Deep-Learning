import numpy as np
import matplotlib.pyplot as plt

n_vectors = 3
vecs = np.random.randn(2, n_vectors)    # A matrix of shape (2, 3)

print(vecs)

n_coeffs = n_vectors    # Must be the number of dimensions. Yeah: the product of the dimensions.

# 32 x 32 x 3 = 3072 coefficients.
# Christ. How the fuck did they get away with just adding them?

# Random numbers between 0 and 1. Don't confuse with randn.
coeffs = np.random.rand(n_coeffs, 10000)

sum_of_first_two = np.sum(coeffs[:2], axis=0)   # Why the first two?

# This isn't very helpful. How do I generalise it to n dimensions?

coeffs[2] = 1 - sum_of_first_two # (*)

# First two coefficients guaranteed to be positive
# Third coefficcient could be negative
# All coefficients guaranteed to add up to one because of (*)
good_coeffs = coeffs[2] >= 0    # Find those greater than 0.
print(good_coeffs)

# Mine are, in a sense, coefficients. The dimensionality doesn't matter.
#

coeffs = coeffs[:, good_coeffs]




