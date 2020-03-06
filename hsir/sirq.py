from collections import namedtuple
import numpy as np

import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.optimize import minimize
from easymh import mh

from .law import Bin, Poi


__all__ = ['SIRQ', 'InferSIRQ']


class SIRQ:
    def __init__(self, beta, gamma, theta, dt=1):
        self.beta = beta * dt
        self.gamma = gamma * dt
        self.theta = theta * dt
        
    def __repr__(self):
        return "β={}, γ={}, θ={}".format(self.beta, self.gamma, self.theta)

    def r0(self):
        return self.beta / (self.gamma + self.theta)

    def estimate(self, region, T):
        S = np.zeros(T+1)
        I = np.zeros(T+1)
        R = np.zeros(T+1)
        Q = np.zeros(T+1)
        
        S[0] = region.S
        I[0] = region.I
        R[0] = region.R
        Q[0] = region.Q
        N = S[0] + I[0] + R[0] + Q[0]

        for t in range(T):
            a, b, c = self.beta*S[t]*I[t]/N, self.gamma*I[t], self.theta*I[t]
            S[t+1] = S[t] - a
            I[t+1] = I[t] + a - b - c
            R[t+1] = R[t] + b
            Q[t+1] = Q[t] + c
        
        Epidemic = namedtuple('Epidemic', 'S I R Q')
        return Epidemic(S, I, R, Q)

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
        fig.add_scatter(x=np.arange(T+1), y=epidemic.Q.astype(int), name="Quarantined", hovertemplate="%{y}")
        return fig  


def loglikely(epidemic, sample, law_s, confirmed, law_c, weight_c=1):
    ll = 0

    if sample is not None:
        ms = sample.m
        ns = sample.n
        ds = epidemic.I[sample.t] / ms
        ks = sample.positive
        ll = sum(law_s.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks))

    if confirmed is not None:
        qs = epidemic.Q[confirmed.t]
        cs = confirmed.c
        ll += weight_c * sum(law_c.loglikely(n, 1, k) for n, k in zip(qs, cs))

    return ll


def likelihood(epidemic, sample, law_s, confirmed, law_c, weight_c=1):
    return np.exp(loglikely(epidemic, sample, law_s, confirmed, law_c, weight_c))


class InferSIRQ():
    def __init__(self, law_s=Bin, law_c=Poi, algo="map", weight_c=1):
        self.law_s = law_s
        self.law_c = law_c
        self.algo = algo
        self.weight_c = weight_c
        
    def __str__(self):
        return "β={}, γ={}, θ={}, loglikely={}".format(self.beta, self.gamma, self.theta, self.loglikely)
    
    def plot(self, beta, region, sample, confirmed, law_s=None, law_c=None, weight_c=None):
        if law_s is None:
            law_s = self.law_s
        if law_c is None:
            law_c = self.law_c
        if weight_c is None:
            weight_c = self.weight_c

        x, y = np.logspace(-2, -0.3, 50), np.logspace(-2, -0.3, 50)
        z = np.zeros((len(y), len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                dynamic = SIRQ(beta, x[j], y[i])
                epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
                z[i, j] = loglikely(epidemic, sample, law_s, confirmed, law_c, weight_c) 

        fig = go.Figure(data=go.Contour(z=np.log(np.max(z)-z+1), x=x, y=y, showscale=False, name='log(-loglikely)'))
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=25),
            xaxis=dict(scaleanchor="y", scaleratio=1, constrain="domain", range=(-2, -0.3))
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        return fig
    
    def plot_3d(self, region, sample, confirmed, law_s=None, law_c=None, weight_c=None):
        if law_s is None:
            law_s = self.law_s
        if law_c is None:
            law_c = self.law_c
        if weight_c is None:
            weight_c = self.weight_c
            
        a, b, c = np.logspace(-2, 0, 20), np.logspace(-2, -0.3, 15), np.logspace(-2, -0.3, 15)
#        d = np.zeros((len(a), len(b), len(c)))
#        for i in range(len(a)):
#            for j in range(len(b)):
#                for k in range(len(c)):
#                    dynamic = SIRQ(a[i], b[j], c[k])
#                    epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
#                    d[i, j, k] = loglikely(epidemic, confirmed, sample, law)
        
        l = len(a) * len(b) * len(c)
        x = np.zeros(l)
        y = np.zeros(l)
        z = np.zeros(l)
        d = np.zeros(l)
        i = 0
        for beta in a:
            for gamma in b:
                for theta in c:
                    x[i] = beta
                    y[i] = gamma
                    z[i] = theta
                    dynamic = SIRQ(beta, gamma, theta)
                    epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
                    d[i] = loglikely(epidemic, sample, law_s, confirmed, law_c, weight_c) 
                    i += 1
        
        fig = px.scatter_3d(x=x, y=y, z=z, color=np.log(np.max(d)-d+1), labels='log(-loglikely)',
                            log_x=True, log_y=True, log_z=True, opacity=0.8,
                            color_continuous_scale=px.colors.sequential.Oranges_r)
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=25),
        )
        return fig

    def fit(self, region, sample, confirmed, law_s=None, law_c=None, weight_c=None, algo=None, **kvarg):
        if law_s is None:
            law_s = self.law_s
        if law_c is None:
            law_c = self.law_c
        if weight_c is None:
            weight_c = self.weight_c
        if algo is None:
            algo = self.algo
            
        if algo == "map":
            self.fit_beta_gamma_map(region, sample, confirmed, law_s, law_c, weight_c, **kvarg)
        elif algo == "mcmc":
            self.fit_beta_gamma_mh(region, sample, confirmed, law_s, law_c, weight_c, **kvarg)   
            
    def fit_beta_gamma_map(self, region, sample, confirmed, law_s=None, law_c=None, weight_c=None, **kvarg):
        if law_s is None:
            law_s = self.law_s
        if law_c is None:
            law_c = self.law_c
        if weight_c is None:
            weight_c = self.weight_c            
            
        def func(x):
            dynamic = SIRQ(*x)
            epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
            return -loglikely(epidemic, sample, law_s, confirmed, law_c, weight_c)
        
        res = minimize(func, (0.5, 0.25, 0.25), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        self.beta, self.gamma, self.theta = res.x
        self.loglikely = -res.fun
        fig = self.plot(self.beta, region, sample, confirmed, law_s, law_c, weight_c)
        fig.add_scatter(x=[self.gamma], y=[self.theta], name='optimum')
        fig.show()
        
    def fit_beta_gamma_mh(self, region, sample, confirmed, law_s=None, law_c=None, weight_c=None, method='naive', **kvarg):
        if law_s is None:
            law_s = self.law_s
        if law_c is None:
            law_c = self.law_c
        if weight_c is None:
            weight_c = self.weight_c 
                
        def func(x):
            dynamic = SIRQ(*x)
            epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
            like = likelihood(epidemic, sample, law_s, confirmed, law_c, weight_c)
            return like
        
        def func2(x):
            dynamic = SIRQ(*np.exp(x))
            epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
            return likelihood(epidemic, sample, law_s, confirmed, law_c, weight_c) * np.prod(np.exp(x))
            
        if method == 'naive':
            x, walker, like = mh([0.5, 0.1, 0.1], func, [[0.01, 1], [0.01, 1], [0.01, 1]], **kvarg)
        elif method == 'mirror':
            x, walker, like = mh([0.5, 0.1, 0.1], func, [[0.01, 1], [0.01, 1], [0.01, 1]], ascdes=(np.log, np.exp), **kvarg)
        elif method == 'repar':
            x, walker, like = mh(np.log([0.5, 0.1, 0.1]), func2, np.log([[0.01, 1], [0.01, 1], [0.01, 1]]), **kvarg)
            x = np.exp(x)
            walker = np.exp(walker)
            like /= np.prod(walker, axis=1)

        self.beta, self.gamma, self.theta = x
        self.loglikely = np.log(func(x))
        self.walker = walker
        self.like = like

        fig = px.line_3d(x=self.walker[:, 0], y=self.walker[:, 1], z=self.walker[:, 2],
                         log_x=True, log_y=True, log_z=True)
        fig.show()  
