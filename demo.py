#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from hsir.law import Bin, Poi
from hsir.empirical import Region, Sample, Confirmed
from hsir.sir import SIR, InferSIR
from hsir.sirq import SIRQ, InferSIRQ


# In[2]:


city = Region(990, 10, 0, 0)
city


# In[9]:


dynamic = SIR(3, 1, 0.1)


# In[10]:


T = 100
epi = dynamic.predict(city, T)


# In[16]:


sample = Sample(epi, np.arange(T//10, T, T//10), 1000*np.ones(9), 10*np.ones(9), Poi, seed=0)


# In[17]:


fig = SIR.plot(epi)
sample.plot(fig)


# In[24]:


a = InferSIR(law=Bin)


# In[25]:


a.plot(city, sample)


# In[27]:


a.fit(city, sample, algo='map')
a.beta, a.gamma, a.loglikely


# In[29]:


a.fit(city, sample, algo='mcmc', method='naive')
a.beta, a.gamma, a.loglikely


# In[30]:


a.fit(city, sample, algo='mcmc', method='mirror')
a.beta, a.gamma, a.loglikely


# In[31]:


a.fit(city, sample, algo='mcmc', method='repar')
a.beta, a.gamma, a.loglikely


# In[32]:


dynamic = SIRQ(3, 0.3, 0.7, 0.1)


# In[33]:


T = 100
epi = dynamic.predict(city, T)


# In[35]:


sample = Sample(epi, np.arange(T//10, T, T//10), 1000*np.ones(9), 10*np.ones(9), Poi, seed=0)
confirmed = Confirmed(epi, np.arange(T//15, T//2, T//15), seed=0)


# In[37]:


fig = SIRQ.plot(epi)
sample.plot(fig)
confirmed.plot(fig)


# In[38]:


a = InferSIRQ(law=Bin)


# In[39]:


a.plot_3d(city, confirmed, sample)


# In[41]:


a.fit(city, confirmed, sample, algo='map')
a.beta, a.gamma, a.theta, a.loglikely


# In[43]:


a.fit(city, confirmed, sample, algo='mcmc', method='naive', N=1000)
a.beta, a.gamma, a.theta, a.loglikely


# In[44]:


a.fit(city, confirmed, sample, algo='mcmc', method='mirror', N=1000)
a.beta, a.gamma, a.theta, a.loglikely


# In[45]:


a.fit(city, confirmed, sample, algo='mcmc', method='repar', N=1000)
a.beta, a.gamma, a.theta, a.loglikely


# In[ ]:




