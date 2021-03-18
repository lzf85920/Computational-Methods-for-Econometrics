#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression

np.random.seed(0)

u1 = np.random.gumbel(0, 1, 400)
u2 = np.random.gumbel(0, 1, 400)
x1 = np.random.normal(0, 1, 400)
x2 = np.random.chisquare(1, 400)

beta1 = np.zeros(400) + 1
beta2 = np.zeros(400) - 0.5

y = ((x1 * beta1) + u1) - ((x2 * beta2) + u2)
y[y > 0] = 1
y[y < 0] = 0

def grid_search(n):
    beta_1 = np.array(np.arange(-5, 5, n))
    beta_2 = np.array(np.arange(-5, 5, n))

    f_max = -1000
    
    for i in beta_1:
        for j in beta_2:
            G = np.exp(x1*i - x2*j) / (1 + np.exp(x1*i - x2*j))
            f = (y*np.log(G) + (1 - y)*np.log(1 - G)).sum()
             
            if f > f_max:
                f_max = f
                
                beta1_hat = i
                beta2_hat = j


    return beta1_hat, beta2_hat
  
start = time.time()
beta1, beta2 = grid_search(0.025)
end = time.time()

print('Beta_1 : ' , beta1)
print('Beta_2 : ' , beta2)
print('Time : ', end - start)

def data_generation():
    u1 = np.random.gumbel(0, 1, 400)
    u2 = np.random.gumbel(0, 1, 400)
    x1 = np.random.normal(0, 1, 400)
    x2 = np.random.chisquare(1, 400)

    beta1 = np.zeros(400) + 1
    beta2 = np.zeros(400) - 0.5
    
    y = ((x1 * beta1) + u1) - ((x2 * beta2) + u2)
    y[y > 0] = 1
    y[y < 0] = 0
    X = np.hstack((x1.reshape(-1,1), -x2.reshape(-1,1)))
    return X, y
  
coef = pd.DataFrame()

for i in range(100):
    X, y = data_generation()
    clf = LogisticRegression(solver = 'lbfgs').fit(X, y)
    coe = pd.DataFrame(clf.coef_)
    coef = coef.append(coe)
    
beta1_mean = coef[0].mean()
beta1_std = coef[0].std()

beta2_mean = coef[1].mean()
beta2_std = coef[1].std()

print('beta1_mean : ', beta1_mean)
print('beta1_std : ', beta1_std)
print('beta2_mean : ', beta2_mean)
print('beta2_std : ', beta2_std)
