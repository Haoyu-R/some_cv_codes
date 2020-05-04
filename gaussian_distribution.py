from math import *
import numpy as np


def gaussian(x, sigma2, mu):
    return 1 / sqrt(2. * np.pi * sigma2) * exp(-0.5 * (x - mu) ** 2 / sigma2)


def update(mu_1, sigma2_1, mu_2, sigma2_2):
    return (sigma2_1 * mu_2 + sigma2_2 * mu_1) / (sigma2_1 + sigma2_2), 1 / (1 / sigma2_2 + 1 / sigma2_1)


def predict(mu_1, sigma2_1, mu_2, sigma2_2):
    return mu_2 + mu_1, sigma2_1 + sigma2_2


measurements = [5, 6, 7, 9, 10]
motion = [1, 1, 2, 1, 1]
measurement_sig = 4
motion_sig = 2
mu = 0
sig = 0.0001

for i in range(len(measurements)):
    mu, sig = update(mu, sig, measurements[i], measurement_sig)
    print(mu, sig)
    mu, sig = predict(mu, sig, motion[i], motion_sig)
    print(mu, sig)
