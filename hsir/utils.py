import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots


__all__ = ['Id', 'JumpProcess']


def Id(x, *vargs, **kvargs):
    return x


class JumpProcess:
    @staticmethod
    def jump(start, amplitude, wait, horizon, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        process = np.zeros(horizon)
        t = 0
        while t < horizon:
            T = int(np.random.exponential(wait))
            process[t:t+T] = start
            start *= (1 + np.random.choice([-1, 1]) * amplitude)
            t += T
        return process

    @staticmethod
    def pulse(base, wait, high, duration, horizon, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        process = np.zeros(horizon)
        t = 0
        while t < horizon:
            T = int(np.random.exponential(wait))
            process[t:t+T] = base
            t += T
            T = int(np.random.exponential(duration))
            process[t:t+T] = high
            t += T
        return process        
    
    @staticmethod
    def plot(y, log=True):
        fig = px.scatter(y=y)
        fig.update_layout(margin=dict(b=0, l=0, r=150, t=25))
        if log:
            fig.update_yaxes(type="log")
        return fig
