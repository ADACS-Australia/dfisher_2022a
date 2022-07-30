import numpy as np

def generate_1gaussian_test_data(mu, sigma, npoints, height=1, offset=0):
    y = np.random.normal(mu, sigma, npoints)*height + offset
    return y

def 