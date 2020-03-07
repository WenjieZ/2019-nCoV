import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.optimize import minimize
from easymh import mh

from .law import Bin, Poi, Gau
from .norm import variation1, variation2, elastic_net
from .empirical import Region, Epidemic


__all__ = ['SIRt', 'InferSIRt']


class JumpProcess:
    def __init__(self, start, amplitude, wait, horizon, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        process = np.zeros(horizon)
        t = 0
        while t < horizon:
            T = int(np.random.exponential(wait))
            process[t:t+T] = start
            start *= (1 + np.random.choice([-1, 1]) * amplitude)
            t += T
            
        self.value = process
            
    def plot(self):
        fig = px.scatter(x=range(len(self.value)), y=self.value, hovertemplate="%{y}", line_shape='hv')
        fig.update_yaxes(type="log")
        return fig


class SIRt:
    def __init__(self, beta, gamma, dt=1):
        if np.isscalar(gamma):
            gamma = gamma * np.ones_like(beta)
        elif len(beta) != len(gamma):
            raise Exception("Dimensions not equal.")
            
        self.beta = beta * dt
        self.gamma = gamma * dt
        
    def __repr__(self):
        fig = go.Figure()
        fig.add_scatter(x=np.arange(len(self.beta)), y=self.beta, line_shape='hv', name='β', hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(len(self.gamma)), y=self.gamma, line_shape='hv', name='γ', hovertemplate="%{y}")
        fig.update_yaxes(type="log")
        fig.show()
        return ""
    
    def r0(self):
        return self.beta / self.gamma
    
    def estimate(self, region):
        T = len(self.beta)
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R
        
        for t in range(T):
            M = S[t] + I[t] + R[t]
            a, b = self.beta[t]*S[t]*I[t]/M, self.gamma[t]*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b
            R[t+1] = R[t] + b        
        
        return Epidemic(S, I, R, None)

    def predict(self, region, T):
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R
        
        for t in range(T):
            M = S[t] + I[t] + R[t]
            a, b = self.beta[-1]*S[t]*I[t]/M, self.gamma[-1]*I[t]   # replaced `t` with `-1`
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b
            R[t+1] = R[t] + b        
        
        return Epidemic(S, I, R, None)
    
    @staticmethod
    def plot(epidemic, T0=0, T=None, line=None, fig=None):
        if fig is None:
            fig = go.Figure()
            fig.update_layout(margin=dict(b=0, l=0, r=0, t=25))
        if T is None:    
            T = len(epidemic.S) - 1 + T0
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.S.astype(int), name="Susceptible", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.I.astype(int), name="Infectious", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.R.astype(int), name="Removed", hovertemplate="%{y}")
        return fig
    
    def project(self, region, T):
        epidemic = self.estimate(region)
        fig = self.plot(epidemic, T0=0)
        region = Region(epidemic.S[-1], epidemic.I[-1], epidemic.R[-1], None)
        epidemic = self.predict(region, T)
        self.plot(epidemic, T0=len(self.beta), fig=fig)
        return fig


def loglikely(epidemic, sample, law):
    ms = sample.m
    ns = sample.n
    ds = epidemic.I[sample.t] / ms
    ks = sample.positive
    return sum(law.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks)) / len(sample.t)


def likelihood(epidemic, sample, law):
    return np.exp(loglikely(epidemic, sample, law))


class InferSIRt:
    def __init__(self, law_s=Bin, penalty_b=variation1, weight_b=1, algo='map'):
        self.law_s = law_s
        self.penalty_b = penalty_b
        self.weight_b = weight_b
        self.algo = algo
        self.dynamic = None
        self.loglikely = None
        
    def __str__(self):
        fig = go.Figure()
        fig.add_scatter(x=np.arange(len(self.beta)), y=self.beta, line_shape='hv', name='β', hovertemplate="%{y}")
        fig.update_yaxes(type="log")
        fig.show()
        return "γ={}, loglikely={}".format(self.gamma, self.loglikely)

    def fit(self, region, sample, **kvarg):
        if self.algo == "map":
            self.fit_beta_gamma_map(region, sample, **kvarg)
        elif self.algo == "mcmc":
            self.fit_beta_gamma_mh(region, sample, **kvarg)
            
    def fit_beta_gamma_map(self, region, sample, **kvarg):
        def func(x):
            dynamic = SIRt(x[1:], x[0])
            epidemic = dynamic.estimate(region)
            ll = -loglikely(epidemic, sample, self.law_s)
            return ll + self.weight_b * self.penalty_b(x[1:])
        
        x0 = 0.2 * np.ones(sample.t[-1]+1)
        res = minimize(func, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True, 'maxiter': 200000})
        self.dynamic = SIRt(res.x[1:], res.x[0])
        self.loglikely = -res.fun

    def fit_beta_gamma_mh(self, region, sample, method='naive', **kvarg):
        def func(x):
            dynamic = SIRt(x[1:], x[0])
            epidemic = dynamic.estimate(region)
            l = likelihood(epidemic, sample, self.law_s)
            return l * np.exp(-self.weight_b * self.penalty_b(x[1:]))
        
        def func2(x):
            x = np.exp(x)
            dynamic = SIRt(x[1:], x[0])
            epidemic = dynamic.estimate(region)
            l = likelihood(epidemic, sample, self.law_s) * np.prod(x)
            return l * np.exp(-self.weight_b * self.penalty_b(x[1:]))
            
        T = sample.t[-1]
        if method == 'naive':
            x, walker, like = mh([0.2]*(T+1), func, np.tile([0.01, 1], (T+1, 1)), **kvarg)
        elif method == 'mirror':
            x, walker, like = mh([0.2]*(T+1), func, np.tile([0.01, 1], (T+1, 1)), ascdes=(np.log, np.exp), **kvarg)
        elif method == 'repar':
            x, walker, _ = mh(np.log([0.2]*(T+1)), func2, np.tile(np.log([0.01, 1]), (T+1, 1)), **kvarg)
            x = np.exp(x)
            walker = np.exp(walker)
            like = np.array([func(x) for x in walker])

        self.dynamic = SIRt(x[1:], x[0])
        self.loglikely = np.log(func(x))
        self.walker = walker
        self.like = like
