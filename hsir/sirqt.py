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


__all__ = ['SIRQt', 'InferSIRQt']


class SIRQt:
    def __init__(self, beta, gamma, theta, dt=1):
        beta = np.array(beta)
        if np.isscalar(gamma):
            gamma = gamma * np.ones_like(beta)
        else:
            gamma = np.array(gamma)
        theta = np.array(theta)
        if not beta.size == gamma.size == theta.size:
            raise Exception("Dimensions not equal. β: {}, γ: {}, θ: {}".format(beta.size, gamma.size, theta.size))
            
        self.beta = beta * dt
        self.gamma = gamma * dt
        self.theta = theta * dt
    
    def self_plot(self, ls='hv', log=False, **kvargs):
        fig = go.Figure()
        fig.add_scatter(x=np.arange(len(self.beta)), y=self.beta, line_shape=ls, name='β', hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(len(self.gamma)), y=self.gamma, line_shape=ls, name='γ', hovertemplate="%{y}")
        fig.add_scatter(x=np.arange(len(self.theta)), y=self.theta, line_shape=ls, name='θ', hovertemplate="%{y}")
        fig.update_layout(margin=dict(b=0, l=0, r=150, t=25))
        if log:
            fig.update_yaxes(type="log")
        return fig
    
    def __repr__(self):
        self.self_plot().show()
        return ""

    def r0(self, control=False):
        if control:
            return self.beta / (self.gamma + self.theta)
        else:
            return self.beta / self.gamma
        
    def estimate(self, region, horizon=None):
        T = self.beta.size if horizon is None else horizon
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        Q = np.zeros(T+1)
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R
        Q[0] = region.Q
        
        for t in range(T):
            if t < self.beta.size:
                beta, gamma, theta = self.beta[t], self.gamma[t], self.theta[t]
            else:
                beta, gamma, theta = self.beta[-1], self.gamma[-1], self.theta[-1]
                
            M = S[t] + I[t] + R[t] + Q[t]
            a, b, c = beta*S[t]*I[t]/M, gamma*I[t], theta*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b - c
            R[t+1] = R[t] + b 
            Q[t+1] = Q[t] + c
        
        return Epidemic(S, I, R, Q)
    
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
        fig.add_scatter(x=np.arange(T0, T+1), y=epidemic.Q.astype(int), name="Quarantined", hovertemplate="%{y}")
        return fig


def loglikely(epidemic, sample=None, resisted=None, confirmed=None, 
              law_s=Bin, law_r=Bin, law_c=Poi, weight_r=1, weight_c=1, *vargs, verbose=False, **kvargs):
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
        ll += weight_r * sum(law_r.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks)) / len(ns)
    if verbose:
        print('resisted:', ll)
        
    if confirmed is not None:
        qs = epidemic.Q[confirmed.t]
        cs = confirmed.c
        ll += weight_c * sum(law_c.loglikely(n, 1, k) for n, k in zip(qs, cs)) / len(qs)
    if verbose:
        print('confirmed:', ll)

    return ll


def likelihood(epidemic, sample=None, resisted=None, confirmed=None, 
              law_s=Bin, law_r=Bin, law_c=Poi, weight_r=1, weight_c=1, *vargs, **kvargs):
    return np.exp(loglikely(epidemic, sample, resisted, confirmed, 
                            law_s, law_r, law_c, weight_r, weight_c, *vargs, **kvargs))


class InferSIRQt:
    def __init__(self, dynamic=None, law_s=Bin, law_r=Bin, weight_r=1, law_c=Poi, weight_c=1, 
                 penalty_b=variation1, weight_b=1, penalty_t=variation1, weight_t=1, 
                 solver=minimize, maxiter=None, ascdes=(Id, Id), **options):
        self.dynamic = dynamic
        self.law_s = law_s
        self.law_r = law_r
        self.law_c = law_c
        self.weight_r = weight_r
        self.weight_c = weight_c
        self.penalty_b = penalty_b
        self.weight_b = weight_b
        self.penalty_t = penalty_t
        self.weight_t = weight_t
        self.ascdes = ascdes
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

    def fit(self, region, sample=None, resisted=None, confirmed=None):
        asc, des = self.ascdes

        def fun(x):
            x = des(x)
            d = len(x) // 2 + 1
            dynamic = SIRQt(x[1:d], x[0], x[d:])
            epidemic = dynamic.estimate(region, d-1)
            ll = -loglikely(epidemic, sample, resisted, confirmed, 
                            self.law_s, self.law_r, self.law_c, self.weight_r, self.weight_c)
            ll += self.weight_b * self.penalty_b(x[1:d])
            ll += self.weight_t * self.penalty_t(x[d:])
            return ll

        if self.dynamic is None:
            T = 0
            if sample is not None:
                T = max(T, sample.t[-1])
            if resisted is not None:
                T = max(T, resisted.t[-1])
            if confirmed is not None:
                T = max(T, confirmed.t[-1])
            x0 = 0.2 * np.ones(1 + 2 * T)
        else:
            x0 = np.hstack((self.dynamic.gamma[0], self.dynamic.beta, self.dynamic.theta))
        self.res = self.solver(fun, asc(x0), **self.options)
        x = des(self.res.x)
        d = len(x) // 2 + 1
        self.dynamic = SIRQt(x[1:d], x[0], x[d:])
        return self
            
    def fit_beta_gamma_theta_map(self, region, sample, confirmed, **options):        
        asc, des = self.ascdes

        def func(x):
            x = des(x)
            d = len(x) // 2 + 1
            dynamic = SIRQt(x[1:d], x[0], x[d:])
            epidemic = dynamic.estimate(region, d-1)
            ll = -loglikely(epidemic, sample, resisted, confirmed, 
                            self.law_s, self.law_r, self.law_c, self.weight_r, self.weight_c, **options)
            ll += self.weight_b * self.penalty_b(x[1:d])
            ll += self.weight_t * self.penalty_t(x[d:])
            return ll
        
        x0 = np.ones(1 + 2 * max(confirmed.t[-1], sample.t[-1]))
        d = len(x0) // 2 + 1
        x0[0] = 0.04
        x0[1:d] = 0.35
        x0[d:] = 0.07
        res = minimize(func, asc(x0), method='nelder-mead', options={'xatol': 1e-8, 'disp': True, 'maxiter': 1000000})
#        res = dual_annealing(func, np.tile([0, 1], (len(x0), 1)), maxiter=100000, seed=0)
        res.x = des(res.x)
        self.dynamic = SIRQt(res.x[1:d], res.x[0], res.x[d:])
        self.funcval = res.fun
