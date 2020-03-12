import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.optimize import minimize
from easymh import mh

from .utils import Id
from .empirical import Region, Epidemic
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
        
        for t in range(T):
            M = S[0] + I[0] + R[0]
            a, b = self.beta*S[t]*I[t]/M, self.gamma*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b
            R[t+1] = R[t] + b        
        
        return Epidemic(S, I, R, None)

    def predict(self, region, T):
        return self.estimate(region, T)
    
    @staticmethod
    def plot(epidemic):
        fig = go.Figure()
        fig.update_layout(margin=dict(b=0, l=0, r=150, t=25))
        T = epidemic.S.size - 1
        fig.add_scatter(x=np.arange(T+1), y=epidemic.S.astype(int), name="Susceptible", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T+1), y=epidemic.I.astype(int), name="Infectious", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T+1), y=epidemic.R.astype(int), name="Removed", hovertemplate="%{y}")
        return fig        


def loglikely(epidemic, sample, law):
    ms = sample.m
    ns = sample.n
    ds = epidemic.I[sample.t] / ms
    ks = sample.positive
    return sum(law.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks)) / len(ns)


def likelihood(epidemic, sample, law):
    return np.exp(loglikely(epidemic, sample, law))


class InferSIR():
    def __init__(self, dynamic=SIR(0.5, 0.1), law=Poi, solver=minimize, maxiter=None, **options):
        self.dynamic = dynamic
        self.law = law
        self.solver = solver
        if solver == minimize and 'method' not in options:
            options['method'] = 'nelder-mead'
        options['options'] = {'maxiter': 100000} if maxiter is None else {'maxiter': maxiter}
        options['options']['disp'] = True
        self.options = options
        
    def plot(self, region, sample):
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

    def fit(self, region, sample, viz=False):
        def fun(x):
            dynamic = SIR(*x)
            epidemic = dynamic.estimate(region, sample.t[-1])
            return -loglikely(epidemic, sample, self.law)

        self.res = self.solver(fun, (self.dynamic.beta, self.dynamic.gamma), **self.options)
        self.dynamic = SIR(*self.res.x)
        if viz:
            fig = self.plot(region, sample)
            fig.add_scatter(x=[self.dynamic.beta], y=[self.dynamic.gamma], name='optimum')
            fig.show()            
        return self
            
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
               
        def func(x):
            dynamic = SIR(*x)
            epidemic = dynamic.estimate(region, sample.t[-1])
            return likelihood(epidemic, sample, law)
        
        def func2(x):
            dynamic = SIR(*np.exp(x))
            epidemic = dynamic.estimate(region, sample.t[-1])
            return likelihood(epidemic, sample, law) * np.prod(np.exp(x))
            
        if method == 'naive':
            x, walker, like = mh([0.5, 0.5], func, [[0.01, 1], [0.01, 1]], **kvarg)
        elif method == 'mirror':
            x, walker, like = mh([0.5, 0.5], func, [[0.01, 1], [0.01, 1]], ascdes=(np.log, np.exp), **kvarg)
        elif method == 'repar':
            x, walker, like = mh(np.log([0.5, 0.5]), func2, np.log([[0.01, 1], [0.01, 1]]), **kvarg)
            x = np.exp(x)
            walker = np.exp(walker)
            like /= np.prod(walker, axis=1)

        self.beta, self.gamma = x
        self.loglikely = np.log(func(x))
        self.walker = walker
        self.like = like

        fig = self.plot(region, sample, law)
        fig.add_scatter(x=self.walker[:, 0], y=self.walker[:, 1], mode="markers+lines")
        fig.show()  

