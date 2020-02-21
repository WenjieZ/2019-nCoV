import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from .model import SIR, Epidemic
from .law import Poi

__all__ = ['InferSIR']


def loglikely(epidemic, sample, law):
    ns = sample.n
    ds = epidemic.I[sample.t] / epidemic.N
    ks = sample.positive
    return sum(law.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks))


def likelihood(epidemic, sample, law):
    return np.exp(loglikely(epidemic, sample, law))


identity = lambda x: x


one = lambda x: 1


class InferSIR():
    def __init__(self, law=Poi, algo="map"):
        self.law = law
        self.algo = algo
        
    def __str__(self):
        return "β={}, γ={}, loglikely={}".format(self.beta, self.gamma, self.loglikely)
    
    def plot(self, region, sample, law=None):
        if law is None:
            law = self.law
                    
        x, y = np.logspace(-2, 0, 30), np.logspace(-2, 0, 30)
        z = np.zeros((len(y), len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                dynamic = SIR(x[j], y[i])
                epidemic = Epidemic(region, dynamic, sample.t[-1])
                z[len(x)-1-i, j] = loglikely(epidemic, sample, law)
                
        fig = go.Figure(data=go.Contour(z=np.log(-z), x=x, y=y, showscale=False))
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=25),
            xaxis=dict(scaleanchor="y", scaleratio=1, constrain="domain", range=(-2, 0))
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        return fig

    def fit(self, region, sample, law=None, algo=None, **kvarg):
        if law is None:
            law = self.law
        if algo is None:
            algo = self.algo
            
        if algo == "map":
            self.fit_beta_gamma_map(region, sample, law, **kvarg)
        elif algo == "mcmc":
            self.fit_beta_gamma_mcmc(region, sample, law, **kvarg)      
        
    def fit_beta_gamma_map(self, region, sample, law=None, **kvarg):
        if law is None:
            law = self.law
            
        def func(x):
            dynamic = SIR(x[0], x[1])
            epidemic = Epidemic(region, dynamic, sample.t[-1])
            return -loglikely(epidemic, sample, law)
        
        res = minimize(func, (0.5,0.5), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        self.beta, self.gamma = res.x
        self.loglikely = -res.fun
        fig = self.plot(region, sample, law)
        fig.add_scatter(x=[self.beta], y=[self.gamma])
        fig.show()
        
    def fit_beta_gamma_mcmc(self, region, sample, law=None, N=1000, B=200,
                            mirror=(identity, identity), trans=(identity, one, identity),
                            var=[[0.01, 0],[0, 0.01]], **kvarg):
        if law is None:
            law = self.law
            
        def func(x):
            dynamic = SIR(trans[0](x[0]), trans[0](x[1]))
            epidemic = Epidemic(region, dynamic, sample.t[-1])
            return likelihood(epidemic, sample, law)*trans[1](x[0])*trans[1](x[1])
        
        walker = np.zeros((N+1, 2))
        x = trans[2](np.array([0.5, 0.5]))
        px = func(x)
        walker[0, :] = x
        for n in range(N):
            y = mirror[1](mirror[0](x) + np.random.multivariate_normal((0, 0), var))
            if y[0]<trans[2](0.001) or y[0]>trans[2](1) or y[1]<trans[2](0.001) or y[1]>trans[2](1):
                y = x
            py = func(y)
            if np.random.rand() < py/px:
                x, px = y, py
            walker[n+1, :] = x
            
        res = np.mean(walker[B:, :], axis=0)
        self.loglikely = np.log(func(res)/trans[1](res[0])/trans[1](res[1]))
        res = trans[0](res)
        self.beta = res[0]
        self.gamma = res[1]
        self.walker = trans[0](walker)
                
        fig = self.plot(region, sample, law)
        fig.add_scatter(x=self.walker[0:B, 0], y=self.walker[0:B, 1], mode="markers+lines")
        fig.add_scatter(x=self.walker[B:, 0], y=self.walker[B:, 1], mode="markers")
        fig.show()        
