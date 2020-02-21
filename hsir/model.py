import numpy as np
import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

__all__ = ['Region', 'SIR', 'Epidemic', 'Sample']


class Region:
    def __init__(self, S, I, R):
        self.S = S
        self.I = I
        self.R = R
        self.N = S + I + R
        
    def __repr__(self):
        return "S={}, I={}, R={}".format(self.S, self.I, self.R)

  
class SIR:
    def __init__(self, beta, gamma, dt=1):
        self.beta = beta * dt
        self.gamma = gamma * dt
        
    def __repr__(self):
        return "β={}, γ={}".format(self.beta, self.gamma)

   
class Epidemic:
    def __init__(self, region, dynamic, T):
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R

        for t in range(T):
            a, b = dynamic.beta*S[t]*I[t]/region.N, dynamic.gamma*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b
            R[t+1] = R[t] + b
            
        self.S = S
        self.I = I
        self.R = R
        self.N = region.N
        self.T = T
        
    def plot(self):
        fig = go.Figure()
        fig.update_layout(margin=dict(b=0, l=0, r=0, t=25))
        T = self.T
        fig.add_scatter(x=np.arange(T+1), y=self.S.astype(int), name="Susceptible", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T+1), y=self.I.astype(int), name="Infectious", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T+1), y=self.R.astype(int), name="Removed", hovertemplate="%{y}")
        return fig

   
class Sample:
    def __init__(self, epidemic, ts, ns, law):
        positive = np.zeros_like(ts)
        for i, (t, n) in enumerate(zip(ts, ns)):
            positive[i] = law.sample(n, epidemic.I[t]/epidemic.N)
            
        self.t = ts
        self.n = ns
        self.positive = positive
        self._law = law
        
    def __repr__(self):
        return " t: {} \n n: {} \n positive: {}".format(self.t, self.n, self.positive)
    
    def plot(self, epidemic):
        fig = epidemic.plot()
        fig.add_scatter(
            x=self.t, y=self.positive/self.n*epidemic.N, 
            mode="markers", name="Guessed", hovertemplate="%{y}"
        )
        return fig
