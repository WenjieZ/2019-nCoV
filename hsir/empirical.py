from collections import namedtuple
import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from .law import Poi


__all__ = ['Region', 'Epidemic', 'Sample', 'Confirmed', 'Resisted']


Region = namedtuple('Region', 'S I R Q')
Epidemic = namedtuple('Epidemic', 'S I R Q')


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
            mode="markers", marker_symbol=1, name="Estimated", hovertemplate="%{y}"
        )
        return fig
    

class Confirmed:
    def __init__(self, epidemic, ts, law=Poi, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        c = np.zeros_like(ts)
        for i, t in enumerate(ts):
            c[i] = law.sample(epidemic.Q[t], 1)

        self.t = ts
        self.c = c
        
    def __repr__(self):
        return " t: {} \n c: {}".format(self.t, self.c)
    
    def plot(self, fig):
        fig.add_scatter(
            x=self.t, y=self.c, 
            mode="markers", marker_symbol=2, name="Confirmed", hovertemplate="%{y}"
        )
        return fig
    
    
class Resisted:
    def __init__(self, epidemic, ts, ns, law, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        positive = np.zeros_like(ts)
        ms = np.zeros_like(ts)
        for i, (t, n) in enumerate(zip(ts, ns)):
            ms[i] = epidemic.S[t] + epidemic.I[t] + epidemic.R[t]
            positive[i] = law.sample(n, epidemic.R[t] / ms[i])
            
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
            mode="markers", marker_symbol=3, name="Resisted", hovertemplate="%{y}"
        )
        return fig
