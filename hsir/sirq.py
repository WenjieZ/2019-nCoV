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
    

def loglikely(epidemic, confirmed, sample, law):
    ms = sample.m
    ns = sample.n
    ds = epidemic.I[sample.t] / ms
    ks = sample.positive
    ll = sum(law.loglikely(n, d, k) for n, d, k in zip(ns, ds, ks))
    qs = epidemic.Q[confirmed.t]
    cs = confirmed.c
    ll += sum(-np.log(c/q)**2 for q, c in zip(qs, cs))
    return ll


def likelihood(epidemic, confirmed, sample, law):
    return np.exp(loglikely(epidemic, confirmed, sample, law))


class InferSIRQ():
    def __init__(self, law=Poi, algo="map"):
        self.law = law
        self.algo = algo
        
    def __str__(self):
        return "β={}, γ={}, θ={}, loglikely={}".format(self.beta, self.gamma, self.theta, self.loglikely)
    
    def plot(self, beta, region, confirmed, sample, law=None):
        if law is None:
            law = self.law

        x, y = np.logspace(-2, -0.3, 50), np.logspace(-2, -0.3, 50)
        z = np.zeros((len(y), len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                dynamic = SIRQ(beta, x[j], y[i])
                epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
                z[i, j] = loglikely(epidemic, confirmed, sample, law) 

        fig = go.Figure(data=go.Contour(z=np.log(np.max(z)-z+1), x=x, y=y, showscale=False, name='log(-loglikely)'))
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=25),
            xaxis=dict(scaleanchor="y", scaleratio=1, constrain="domain", range=(-2, -0.3))
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        return fig
    
    def plot_3d(self, region, confirmed, sample, law=None):
        if law is None:
            law = self.law

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
                    d[i] = loglikely(epidemic, confirmed, sample, law) 
                    i += 1
        
        fig = px.scatter_3d(x=x, y=y, z=z, color=np.log(np.max(d)-d+1), labels='log(-loglikely)',
                            log_x=True, log_y=True, log_z=True, opacity=0.8,
                            color_continuous_scale=px.colors.sequential.Oranges_r)
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=25),
        )
        return fig

    def fit(self, region, confirmed, sample, law=None, algo=None, **kvarg):
        if law is None:
            law = self.law
        if algo is None:
            algo = self.algo
            
        if algo == "map":
            self.fit_beta_gamma_map(region, confirmed, sample, law, **kvarg)
        elif algo == "mcmc":
            self.fit_beta_gamma_mh(region, confirmed, sample, law, **kvarg)   
            
    def fit_beta_gamma_map(self, region, confirmed, sample, law=None, **kvarg):
        if law is None:
            law = self.law
            
        def func(x):
            dynamic = SIRQ(*x)
            epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
            return -loglikely(epidemic, confirmed, sample, law)
        
        res = minimize(func, (0.5, 0.25, 0.25), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        self.beta, self.gamma, self.theta = res.x
        self.loglikely = -res.fun
        fig = self.plot(self.beta, region, confirmed, sample, law)
        fig.add_scatter(x=[self.gamma], y=[self.theta], name='optimum')
        fig.show()
        
    def fit_beta_gamma_mh(self, region, confirmed, sample, law=None, method='naive', **kvarg):
        if law is None:
            law = self.law
        
        if 'width' not in kvarg:
            kvarg['width'] = 0.1
        
        def func(x):
            dynamic = SIRQ(*x)
            epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
            return likelihood(epidemic, confirmed, sample, law)
        
        def func2(x):
            dynamic = SIRQ(*np.power(10, x))
            epidemic = dynamic.estimate(region, max(confirmed.t[-1], sample.t[-1]))
            return likelihood(epidemic, confirmed, sample, law) * np.prod(10**x)
            
        if method == 'naive':
            res, walker = mh([0.5, 0.25, 0.25], func, np.array([[0.01, 1], [0.01, 1], [0.01, 1]]), **kvarg)
        elif method == 'mirror':
            res, walker = mh([0.5, 0.25, 0.25], func, np.array([[0.01, 1], [0.01, 1], [0.01, 1]]), ascdes=(np.log, np.exp), **kvarg)
        elif method == 'repar':
            res, walker = mh([-1., -1., -1.], func2, np.array([[-2, 0], [-2, 0], [-2, 0]]), **kvarg)
            res = np.power(10, res)
            walker = np.power(10, walker)

        self.beta, self.gamma, self.theta = res
        self.loglikely = np.log(func(res))
        self.walker = walker

        fig = px.line_3d(x=self.walker[:, 0], y=self.walker[:, 1], z=self.walker[:, 2],
                         log_x=True, log_y=True, log_z=True)
        fig.show()  
