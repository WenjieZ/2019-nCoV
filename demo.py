#!/usr/bin/env python
# coding: utf-8

#  \\[
#  \begin{aligned}
#  \frac{dS}{dt} &= -\frac{\beta SI}{N} \\
#  \frac{dI}{dt} &= \frac{\beta SI}{N} - \gamma I\\
#  \frac{dR}{dt} &= \gamma I 
#  \end{aligned}
#  \\]

# In[1]:


import numpy as np
from hsir.law import Bin, Poi
from hsir.model import Region, SIR, Epidemic, Sample
from hsir.infer import InferSIR


# In[2]:


T = 100
city = Region(9990, 10, 0)
epidemic = Epidemic(city, SIR(3, 1, 0.1), T)
np.random.seed(0)
sample = Sample(epidemic, np.arange(T//10, T, T//10), 10*np.ones(9, dtype=int), Poi)
sample.positive = np.array([0, 1, 2, 6, 1, 0, 0, 1, 0])
sample.plot(epidemic)


# In[3]:


learner = InferSIR(algo="mcmc")
learner.fit(city, sample)
print(learner)

