import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.optimize import minimize
from easymh import mh

from .utils import Id
from .empirical import Region, Epidemic
from .law import Bin, Poi, Gau
from .norm import variation1, variation2, elastic_net


__all__ = ['SIRt', 'InferSIRt']


class SIRt:
    def __init__(self, beta, gamma, dt=1):
        beta = np.array(beta)
        if np.isscalar(gamma):
            gamma = gamma * np.ones_like(beta)
        else:
            gamma = np.array(gamma)
        if beta.size != gamma.size:
            raise Exception("Dimensions not equal. β: {}, γ: {}".format(beta.size, gamma.size))
            
        self.beta = beta * dt
        self.gamma = gamma * dt
        
    def self_plot(self, ls='hv', log=False, **kvargs):
        fig = go.Figure()
        fig.add_scatter(x=np.arange(len(self.beta)), y=self.beta, line_shape=ls, name='β', hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(len(self.gamma)), y=self.gamma, line_shape=ls, name='γ', hovertemplate="%{y}")
        fig.update_layout(margin=dict(b=0, l=0, r=150, t=25))
        if log:
            fig.update_yaxes(type="log")
        return fig
    
    def __repr__(self):
        self.self_plot().show()
        return ""

    def r0(self):
        return self.beta / self.gamma
    
    def estimate(self, region, horizon=None):
        T = self.beta.size if horizon is None else horizon
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R
        
        for t in range(T):
            if t < self.beta.size:
                beta, gamma = self.beta[t], self.gamma[t]
            else:
                beta, gamma = self.beta[-1], self.gamma[-1]
                
            M = S[t] + I[t] + R[t]
            a, b = beta*S[t]*I[t]/M, gamma*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b
            R[t+1] = R[t] + b 
        
        return Epidemic(S, I, R, None)

    @staticmethod
    def plot(epidemic, T0=0, T=None, line=None, fig=None):
        if fig is None:
            fig = go.Figure()
            fig.update_layout(margin=dict(b=0, l=0, r=150, t=25))
        if T is None:    
            T = epidemic.S.size - 1 + T0
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.S.astype(int), name="Susceptible", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.I.astype(int), name="Infectious", hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.R.astype(int), name="Removed", hovertemplate="%{y}")
        return fig
    

def loglikely(epidemic, sample=None, resisted=None, 
              law_s=Bin, law_r=Bin, weight_r=1, *vargs, verbose=False, **kvargs):
    ll = 0

    if sample is not None:
        ms = sample.m
        ns = sample.n
        ds = epidemic.I[sample.t] / ms
        ks = sample.positive
        ll = sum(law_s.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks)) / len(ns)
    if verbose:
        print('sample:', ll)

    if resisted is not None:
        ms = resisted.m
        ns = resisted.n
        ds = epidemic.R[resisted.t] / ms
        ks = resisted.positive
        ll += weight_r * sum(law_s.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks)) / len(ns)
    if verbose:
        print('resisted:', ll)
        
    return ll


def likelihood(epidemic, sample=None, resisted=None, 
              law_s=Bin, law_r=Bin, weight_r=1, *vargs, **kvargs):
    return np.exp(loglikely(epidemic, sample, resisted, law_s, law_r, weight_r, *vargs, **kvargs))


class InferSIRt:
    def __init__(self, dynamic=None, law_s=Bin, law_r=Bin, weight_r=1, penalty_b=variation1, weight_b=1,
                 solver=minimize, maxiter=None, **options):
        self.dynamic = dynamic
        self.law_s = law_s
        self.law_r = law_r
        self.weight_r = weight_r
        self.penalty_b = penalty_b
        self.weight_b = weight_b
        self.solver = solver
        if solver == minimize and 'method' not in options:
            options['method'] = 'nelder-mead'
        options['options'] = {'maxiter': 100000} if maxiter is None else {'maxiter': maxiter}
        options['options']['disp'] = True
        self.options = options
        
    def self_plot(self, **options):
        return self.dynamic.self_plot(**options)
    
    def __repr__(self):
        self.self_plot().show()
        return ""

    def fit(self, region, sample=None, resisted=None):
        def fun(x):
            dynamic = SIRt(x[1:], x[0])
            epidemic = dynamic.estimate(region, sample.t[-1])
            ll = -loglikely(epidemic, sample, resisted, self.law_s, self.law_r, self.weight_r)
            return ll + self.weight_b * self.penalty_b(x[1:])
        
        if self.dynamic is None:
            T = 0
            if sample is not None:
                T = max(T, sample.t[-1])
            if resisted is not None:
                T = max(T, resisted.t[-1])
            x0 = 0.2 * np.ones(1 + 2 * T)
        else:
            x0 = np.hstack((self.dynamic.gamma, self.dynamic.beta))
        self.res = self.solver(fun, x0, **self.options)
        self.dynamic = SIRt(self.res.x[1:], self.res.x[0])
        return self
            
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
