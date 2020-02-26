from collections import namedtuple
import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots


__all__ = ['Region', 'Sample', 'Confirmed']


Region = namedtuple('Region', 'S I R Q')


class Sample:
    def __init__(self, epidemic, ts, ms, ns, law, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        positive = np.zeros_like(ts)
        for i, (t, m, n) in enumerate(zip(ts, ms, ns)):
            positive[i] = law.sample(n, epidemic.I[t]/m)
            
        self.t = ts
        self.m = ms
        self.n = ns
        self.positive = positive
        self._law = law
        
    def __repr__(self):
        return " t: {} \n m: {} \n n: {} \n positive: {}".format(self.t, self.m, self.n, self.positive)
    
    def plot(self, fig):
        fig.add_scatter(
            x=self.t, y=self.positive / self.n * self.m, 
            mode="markers", marker_symbol=1, name="Guessed", hovertemplate="%{y}"
        )
        return fig
    

class Confirmed:
    def __init__(self, epidemic, ts, sigma=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        c = np.zeros_like(ts)
        for i, t in enumerate(ts):
            c[i] = epidemic.Q[t] * (1 + sigma*np.random.randn())
            
        self.t = ts
        self.c = c
        self._sigma = sigma
        
    def __repr__(self):
        return " t: {} \n c: {}".format(self.t, self.c)
    
    def plot(self, fig):
        fig.add_scatter(
            x=self.t, y=self.c, 
            mode="markers", marker_symbol=2, name="Confirmed", hovertemplate="%{y}"
        )
        return fig
