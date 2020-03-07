import numpy as np


__all__ = ['variation1', 'variation2', 'elastic_net']


def variation1(x):
    return np.mean(np.abs(np.diff(x)))


def variation2(x):
    return np.sqrt(np.mean(np.diff(x)**2))


def elastic_net(x, mu=1):
    return (variation1(x) + mu*variation2(x)) / (1+mu)
