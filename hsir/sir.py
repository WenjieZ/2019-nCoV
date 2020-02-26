from collections import namedtuple
import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.optimize import minimize
from easymh import mh

from .law import Bin, Poi


__all__ = ['SIR', 'InferSIR']


class SIR:
    def __init__(self, beta, gamma, dt=1):
        self.beta = beta * dt
        self.gamma = gamma * dt
        
    def __repr__(self):
        return "β={}, γ={}".format(self.beta, self.gamma)
    
    def r0(self):
        return self.beta / self.gamma
    
    def estimate(self, region, T):
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R
        N = S[0] + I[0] + R[0]
        
        for t in range(T):
            a, b = self.beta*S[t]*I[t]/N, self.gamma*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b
            R[t+1] = R[t] + b        
        
        Epidemic = namedtuple('Epidemic', 'S I R')
        return Epidemic(S, I, R)

    def predict(self, region, T):
        return self.estimate(region, T)
    
    @staticmethod
    def plot(epidemic):
        fig = go.Figure()
        fig.update_layout(margin=dict(b=0, l=0, r=0, t=25))
        T = len(epidemic.S) - 1
        fig.add_scatter(x=np.arange(T+1), y=epidemic.S.astype(int), name="Susceptible", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T+1), y=epidemic.I.astype(int), name="Infectious", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T+1), y=epidemic.R.astype(int), name="Removed", hovertemplate="%{y}")
        return fig        


def loglikely(epidemic, sample, law):
    ms = sample.m
    ns = sample.n
    ds = epidemic.I[sample.t] / ms
    ks = sample.positive
    return sum(law.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks))


def likelihood(epidemic, sample, law):
    return np.exp(loglikely(epidemic, sample, law))


class InferSIR():
    def __init__(self, law=Poi, algo="map"):
        self.law = law
        self.algo = algo
        
    def __str__(self):
        return "β={}, γ={}, loglikely={}".format(self.beta, self.gamma, self.loglikely)
    
    def plot(self, region, sample, law=None):
        if law is None:
            law = self.law

        x, y = np.logspace(-2, 0, 50), np.logspace(-2, 0, 50)
        z = np.zeros((len(y), len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                dynamic = SIR(x[j], y[i])
                epidemic = dynamic.estimate(region, sample.t[-1])
                z[i, j] = loglikely(epidemic, sample, law)

        fig = go.Figure(data=go.Contour(z=np.log(np.max(z)-z+1), x=x, y=y, showscale=False, name='log(-loglikely)'))
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
            self.fit_beta_gamma_mh(region, sample, law, **kvarg)   
            
    def fit_beta_gamma_map(self, region, sample, law=None, **kvarg):
        if law is None:
            law = self.law
            
        def func(x):
            dynamic = SIR(*x)
            epidemic = dynamic.estimate(region, sample.t[-1])
            return -loglikely(epidemic, sample, law)
        
        res = minimize(func, (0.5,0.5), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        self.beta, self.gamma = res.x
        self.loglikely = -res.fun
        fig = self.plot(region, sample, law)
        fig.add_scatter(x=[self.beta], y=[self.gamma], name='optimum')
        fig.show()
        
    def fit_beta_gamma_mh(self, region, sample, law=None, method='naive', **kvarg):
        if law is None:
            law = self.law
        
        if 'width' not in kvarg:
            kvarg['width'] = 0.1
        
        def func(x):
            dynamic = SIR(*x)
            epidemic = dynamic.estimate(region, sample.t[-1])
            return likelihood(epidemic, sample, law)
        
        def func2(x):
            dynamic = SIR(*np.power(10, x))
            epidemic = dynamic.estimate(region, sample.t[-1])
            return likelihood(epidemic, sample, law) * np.prod(10**x)
            
        if method == 'naive':
            res, walker = mh([0.5, 0.5], func, np.array([[0.01, 1], [0.01, 1]]), **kvarg)
        elif method == 'mirror':
            res, walker = mh([0.5, 0.5], func, np.array([[0.01, 1], [0.01, 1]]), ascdes=(np.log, np.exp), **kvarg)
        elif method == 'repar':
            res, walker = mh([-1., -1.], func2, np.array([[-2, 0], [-2, 0]]), **kvarg)
            res = np.power(10, res)
            walker = np.power(10, walker)

        self.beta, self.gamma = res
        self.loglikely = np.log(func(res))
        self.walker = walker

        fig = self.plot(region, sample, law)
        fig.add_scatter(x=self.walker[:, 0], y=self.walker[:, 1], mode="markers+lines")
        fig.show()  
        
    def r0(self, biased=True, B=200):
        if biased:
            return self.beta / self.gamma
        else:
            return np.mean(self.walker[B:, 0] / self.walker[B:, 1])
