from abc import ABCMeta, abstractmethod
import numpy as np

__all__ = ['Law', 'Bin', 'Poi', 'Gau']


class Law(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def sample(n, d):
        pass

    @staticmethod
    @abstractmethod
    def loglikely(n, d, k):
        pass
   
    @staticmethod
    def likelihood(n, d, k):
        return np.exp(loglikely(n, d, k))


class Bin(Law):
    def sample(n, d):
        return np.random.binomial(n, d)
    
    def loglikely(n, d, k):
        return k*np.log(d) + (n-k)*np.log(1-d)


class Poi(Law):
    def sample(n, d):
        return np.random.poisson(n*d)
    
    def loglikely(n, d, k):
        return k*np.log(n*d) - n*d + k - k*np.log(1+k) # - np.sum(np.log(np.arange(k)+1))
    

class Gau(Law):
    def sample(n, d=1):
        return n * (1 + 0.1*np.random.randn())
    
    def loglikely(n, d, k):
        return -50 * np.log(k/n)**2 
